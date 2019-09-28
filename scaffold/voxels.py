from scaffold.helpers import dimensions, origin
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from .plotting import plot_voxelize_results
from scipy import ndimage

class VoxelCloud:
    def __init__(self, bounds, voxels, grid_size, map):
        self.bounds = bounds
        self.grid_size = grid_size
        self.voxels = voxels
        self.map = map

    def get_boxes(self):
        return m_grid(self.bounds, self.grid_size)

    @staticmethod
    def create(morphology, N):
        hit_detector, box_data = morphology_detector_factory(morphology)
        bounds, voxels, length, error = voxelize(N, box_data, hit_detector)
        # plot_voxelize_results(bounds, voxels, length, error)
        voxel_map = morphology.get_compartment_map(m_grid(bounds, length), voxels, length)
        if error == 0:
            return VoxelCloud(bounds, voxels, length, voxel_map)
        else:
            raise NotImplementedError("Pick random voxels and distribute their compartments to random neighbours")

class Box(dimensions, origin):
    def __init__(self, dimensions=None, origin=None):
        dimensions.__init__(self, dimensions)
        origin.__init__(self, origin)

def m_grid(bounds, size):
    return np.mgrid[
        bounds[0, 0]:bounds[0, 1]:size,
        bounds[1, 0]:bounds[1, 1]:size,
        bounds[2, 0]:bounds[2, 1]:size
    ]

def voxelize(N, box_data, hit_detector, max_iterations=200, precision_iterations=30):
    # Initialise
    bounds = np.column_stack((box_data.origin - box_data.dimensions / 2, box_data.origin + box_data.dimensions / 2))
    box_length = np.max(box_data.dimensions) # Size of the edge of a cube in the box counting grid
    best_length, best_error = box_length, N # Keep track of our best results so far
    last_box_count, last_box_length = 0., 0. # Keep track of the previous iteration for binary search jumps
    precision_i, i = 0., 0. # Keep track of the iterations
    crossed_treshold = False # Should we consider each next iteration as merely increasing precision?

    # Refine the grid size each iteration to find the right amount of boxes that trigger the hit_detector
    while i < max_iterations and precision_i < precision_iterations:
        i += 1
        if crossed_treshold: # Are we doing these iterations just to increase precision, or still trying to find a solution?
            precision_i += 1
        box_count = 0 # Reset box count
        boxes_x, boxes_y, boxes_z = m_grid(bounds, box_length) # Create box counting grid
        # Create a voxel grid where voxels are switched on if they trigger the hit_detector
        voxels = np.zeros((boxes_x.shape[0], boxes_x.shape[1], boxes_x.shape[2]), dtype=bool)
        # Iterate over all the boxes in the total grid.
        for x_i in range(boxes_x.shape[0]):
            for y_i in range(boxes_x.shape[1]):
                for z_i in range(boxes_x.shape[2]):
                    # Get the lower corner of the query box
                    x = boxes_x[x_i,y_i,z_i]
                    y = boxes_y[x_i,y_i,z_i]
                    z = boxes_z[x_i,y_i,z_i]
                    hit = hit_detector(np.array([x, y, z]), box_length) # Is this box a hit? (Does it cover some part of the object?)
                    voxels[x_i, y_i, z_i] = hit # If its a hit, turn on the voxel
                    box_count += int(hit) # If its a hit, increase the box count
        if last_box_count < N and box_count >= N:
            # We've crossed the treshold from overestimating to underestimating
            # the box_length. A solution is found, but more precise values lie somewhere in between,
            # so start counting the precision iterations
            crossed_treshold = True
        if box_count < N: # If not enough boxes cover the object we should decrease the box length (and increase box count)
            new_box_length = box_length - np.abs(box_length - last_box_length) / 2
        else: # If too many boxes cover the object we should increase the box length (and decrease box count)
            new_box_length = box_length + np.abs(box_length - last_box_length) / 2
        # Store the results of this iteration and prepare variables for the next iteration.
        last_box_length, last_box_count = box_length, box_count
        box_length = new_box_length
        if abs(N - box_count) <= best_error: # Only store the following values if they improve the previous best results.
            best_error, best_length = abs(N - box_count), last_box_length
            best_bounds, best_voxels = bounds, voxels

    # Return best results and error
    return best_bounds, best_voxels, best_length, best_error

def detect_box_compartments(tree, box_origin, box_size):
    '''
        Given a tree of compartment locations and a box, it will return the ids of all compartments in the box

        :param box_origin: The lowermost corner of the box.
    '''
    # Get the outer sphere radius of the cube by taking the length of a diagonal through the cube divided by 2
    search_radius = np.sqrt(np.sum([box_size ** 2 for i in range(len(box_origin))])) / 2
    # Translate the query point to the middle of the box and search within the outer sphere radius.
    return tree.query_radius([box_origin + box_size / 2], search_radius)[0]

def morphology_detector_factory(morphology):
    '''
        Will return a hit detector and outer box required to perform voxelization on the morphology.
    '''
    # Transform the compartment object list into a compartment position 3D numpy array
    tree = morphology.compartment_tree
    compartments = tree.get_arrays()[0]
    n_dimensions = range(compartments.shape[1])
    # Create an outer detection box
    outer_box = Box()
    # The outer box dimensions are equal to the maximum distance between compartments in each of n dimensions
    outer_box.dimensions = np.array([np.max(compartments[:, i]) - np.min(compartments[:, i]) for i in n_dimensions])
    # The outer box origin is in the middle of the outer bounds. (So lowermost point + half of dimensions)
    outer_box.origin = np.array([np.min(compartments[:, i]) + outer_box.dimensions[i] / 2 for i in n_dimensions])
    # Create the detector function
    def morphology_detector(box_origin, box_size):
        # Report a hit if more than 0 compartments are within the box.
        return len(detect_box_compartments(tree, box_origin, box_size)) > 0
    # Return the morphology detector function and box data as the factory products
    return morphology_detector, outer_box

def center_of_mass(points, weights):
    return ndimage.center_of_mass(np.column_stack((points, weights)))
