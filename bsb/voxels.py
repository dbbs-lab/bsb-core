from bsb.helpers import dimensions, origin
import numpy as np
from scipy import ndimage
from time import sleep
from sklearn.neighbors import KDTree
from .functions import get_distances


class VoxelCloud:
    def __init__(self, bounds, voxels, grid_size, map, occupancies=None):
        from rtree import index

        self.bounds = bounds
        self.grid_size = grid_size
        self.voxels = voxels
        self.voxel_cache = None
        self.map = map
        self.occupancies = occupancies
        p = index.Property(dimension=3)
        voxel_tree = index.Index(properties=p)
        voxel_positions = self.get_voxels()
        # Add each voxel box to an Rtree index
        for v, voxel_position in enumerate(voxel_positions):
            av = np.add(voxel_position, grid_size)
            bv = np.concatenate((voxel_position, av))
            voxel_tree.insert(v, tuple(bv))
        self.tree = voxel_tree

    def get_boxes(self):
        return m_grid(self.bounds, self.grid_size)

    def get_voxels(self, cache=False):
        def _voxels():
            return self.get_boxes()[:, self.voxels].T

        if cache:
            if self.voxel_cache is not None:
                return self.voxel_cache
            self.voxel_cache = _voxels()
            return self.voxel_cache
        return _voxels()

    def get_occupancies(self):
        if self.occupancies is None:
            voxel_occupancy = np.array(list(map(lambda x: len(x), self.map)))
            max_voxel_occupancy = max(voxel_occupancy)
            normalized_voxel_occupancy = voxel_occupancy / (max_voxel_occupancy)
            self.occupancies = normalized_voxel_occupancy
        return self.occupancies

    def center_of_mass(self):
        boxes = self.get_boxes()
        points = boxes + self.grid_size / 2
        voxels = self.voxels
        occupancies = self.get_occupancies()
        point_positions = np.column_stack(points[:, voxels]).T
        return center_of_mass(point_positions, occupancies)

    @staticmethod
    def create(morphology, N, compartments=None):
        from rtree import index
        from rtree.index import Rtree

        p = index.Property(dimension=3)
        tree = index.Index(properties=p)
        if compartments is None:
            compartments = morphology.compartments
        N = min(len(compartments), N)
        for compartment in compartments:
            tree.insert(
                int(compartment.id), tuple([*compartment.midpoint, *compartment.midpoint])
            )
        hit_detector = HitDetector.for_rtree(tree)
        bounds, voxels, length, error = voxelize(
            N, morphology.get_bounding_box(compartments=compartments), hit_detector
        )
        voxel_map = morphology.create_compartment_map(
            tree, m_grid(bounds, length), voxels, length
        )
        if error == 0:
            return VoxelCloud(bounds, voxels, length, voxel_map)
        else:
            raise NotImplementedError(
                "Voxelization error: could not find the right amount of voxels. Try N {}".format(
                    # Suggest the closest we got to N for a next attempt
                    ("+" if error > 0 else "")
                    + str(error)
                )
            )

    def intersect(self, other):
        raise NotImplementedError("Intersecting 2 voxel clouds is a to do")

    def get_voxel_box(self):
        """
        Return the box encompassing the voxels.
        """
        voxel_positions = self.get_voxels()
        min_x, min_y, min_z, max_x, max_y, max_z = (
            np.min(voxel_positions[:, 0]),
            np.min(voxel_positions[:, 1]),
            np.min(voxel_positions[:, 2]),
            np.max(voxel_positions[:, 0]),
            np.max(voxel_positions[:, 1]),
            np.max(voxel_positions[:, 2]),
        )
        box = [
            min_x,
            min_y,
            min_z,
            max_x + self.grid_size,
            max_y + self.grid_size,
            max_z + self.grid_size,
        ]
        # print(min_x, min_y, min_z, max_x, max_y, max_z)
        return box


_class_dimensions = dimensions
_class_origin = origin


class Box(dimensions, origin):
    def __init__(self, dimensions=None, origin=None):
        _class_dimensions.__init__(self, dimensions)
        _class_origin.__init__(self, origin)

    @staticmethod
    def from_bounds(bounds):
        dimensions = np.amax(bounds, axis=1) - np.amin(bounds, axis=1)
        origin = np.amin(bounds, axis=1) + dimensions / 2
        return Box(dimensions=dimensions, origin=origin)

    def bounds(self):
        dim = self.dimensions
        dim[dim < 5.0] = 5.0
        # Moving the bounds by a fraction helps prevent points on a plane being
        # sorted into multiple voxels
        return np.column_stack(
            (self.origin - dim / 2 - 0.0001, self.origin + dim / 2 + 0.0004)
        )


def m_grid(bounds, size):
    return np.mgrid[
        bounds[0, 0] : bounds[0, 1] : size,
        bounds[1, 0] : bounds[1, 1] : size,
        bounds[2, 0] : bounds[2, 1] : size,
    ]


def voxelize(N, box_data, hit_detector, max_iterations=80, precision_iterations=30):
    # Initialise
    bounds = box_data.bounds()
    box_length = np.max(
        box_data.dimensions
    )  # Size of the edge of a cube in the box counting grid
    best_length, best_error = box_length, N  # Keep track of our best results so far
    last_box_count, last_box_length = (
        0.0,
        0.0,
    )  # Keep track of the previous iteration for binary search jumps
    precision_i, i = 0.0, 0.0  # Keep track of the iterations
    crossed_treshold = (
        False  # Should we consider each next iteration as merely increasing precision?
    )

    # Refine the grid size each iteration to find the right amount of boxes that trigger the hit_detector
    while i < max_iterations and precision_i < precision_iterations:
        i += 1
        if (
            crossed_treshold
        ):  # Are we doing these iterations just to increase precision, or still trying to find a solution?
            precision_i += 1
        box_count = 0  # Reset box count
        boxes_x, boxes_y, boxes_z = m_grid(bounds, box_length)  # Create box counting grid
        # Create a voxel grid where voxels are switched on if they trigger the hit_detector
        voxels = np.zeros(
            (boxes_x.shape[0], boxes_x.shape[1], boxes_x.shape[2]), dtype=bool
        )
        # Iterate over all the boxes in the total grid.
        for x_i in range(boxes_x.shape[0]):
            for y_i in range(boxes_x.shape[1]):
                for z_i in range(boxes_x.shape[2]):
                    # Get the lower corner of the query box
                    x = boxes_x[x_i, y_i, z_i]
                    y = boxes_y[x_i, y_i, z_i]
                    z = boxes_z[x_i, y_i, z_i]
                    hit = hit_detector(
                        np.array([x, y, z]), box_length
                    )  # Is this box a hit? (Does it cover some part of the object?)
                    voxels[x_i, y_i, z_i] = hit  # If its a hit, turn on the voxel
                    box_count += int(hit)  # If its a hit, increase the box count
        if last_box_count < N and box_count >= N:
            # We've crossed the treshold from overestimating to underestimating
            # the box_length. A solution is found, but more precise values lie somewhere in between,
            # so start counting the precision iterations
            crossed_treshold = True
        if (
            box_count < N
        ):  # If not enough boxes cover the object we should decrease the box length (and increase box count)
            new_box_length = box_length - np.abs(box_length - last_box_length) / 2
        else:  # If too many boxes cover the object we should increase the box length (and decrease box count)
            new_box_length = box_length + np.abs(box_length - last_box_length) / 2
        # Store the results of this iteration and prepare variables for the next iteration.
        last_box_length, last_box_count = box_length, box_count
        box_length = new_box_length
        if (
            abs(N - box_count) <= best_error
        ):  # Only store the following values if they improve the previous best results.
            best_error, best_length = abs(N - box_count), last_box_length
            best_bounds, best_voxels = bounds, voxels

    # Return best results and error
    return best_bounds, best_voxels, best_length, best_error


def detect_box_compartments(tree, box_origin, box_size):
    """
    Given a tree of compartment locations and a box, it will return the ids of all compartments in the outer sphere of the box

    :param box_origin: The lowermost corner of the box.
    """
    # Return all compartment id's that intersect with this box
    return list(
        tree.intersection(tuple([*box_origin, *(box_origin + box_size)]), objects=False)
    )


def center_of_mass(points, weights=None):
    if weights is None:
        cog = [np.sum(points[dim, :]) / points.shape[1] for dim in range(points.shape[0])]
    else:
        cog = [
            np.sum(points[dim, :] * weights) for dim in range(points.shape[0])
        ] / np.sum(weights)
    return cog


def set_attraction(attractor, voxels):
    attraction_voxels = np.indices(voxels.shape)[:, voxels].T
    attraction_map = np.zeros(voxels.shape)
    dist = np.sqrt(
        np.sum(
            (attraction_voxels - attractor + np.ones(len(attractor)) * 0.5) ** 2, axis=1
        )
    )
    distance_sorting = dist.argsort()[::-1]
    attraction = 1
    first_voxel = distance_sorting[0]
    attraction_map[
        attraction_voxels[first_voxel, 0],
        attraction_voxels[first_voxel, 1],
        attraction_voxels[first_voxel, 2],
    ] = 1
    last_distance = dist[first_voxel]
    for v in distance_sorting[1:]:
        distance = dist[v]
        attraction += int(distance < last_distance)
        attraction_map[
            attraction_voxels[v, 0], attraction_voxels[v, 1], attraction_voxels[v, 2]
        ] = attraction
        last_distance = distance
    return attraction_map


class VoxelTransformer:
    def __init__(self, attractor, field):
        self.carriers = []
        self.attractor = attractor
        self.field = field
        self.occupied = {}

    def occupy(self, position):
        if position in self.occupied:
            raise VoxelTransformError("Position already occupied")

    def add_carrier(self, payload, position):
        if position in self.occupied:
            raise VoxelTransformError("Position already occupied")
        carrier = VoxelTransformCarrier(self, payload, position)
        self.carriers.append(carrier)

    def is_unoccupied(self, position):
        return not tuple(position) in self.occupied

    def transform(self):
        ind = np.indices(self.field.shape)[:, self.field > 0].T
        furthest_carrier_first = self.get_furthest_carriers()
        for carrier in furthest_carrier_first:
            dists = np.array(get_distances(ind, carrier.position))
            dist_sort = dists.argsort()
            for attempt in range(len(dist_sort)):
                attempt_position = tuple(ind[dist_sort[attempt]])
                if self.is_unoccupied(attempt_position):
                    self.occupied[attempt_position] = True
                    carrier.position = attempt_position
                    break

    def get_furthest_carriers(self):
        positions = list(map(lambda p: p.position, self.carriers))
        distances = self.get_attractor_distances(positions)
        return np.array(self.carriers)[np.argsort(distances)[::-1]]

    def get_attractor_distances(self, candidates):
        return get_distances(candidates, self.attractor - 0.5)


class VoxelTransformCarrier:
    def __init__(self, transformer, payload, position):
        pos = tuple(position)
        self.transformer = transformer
        self.id = len(transformer.carriers)
        self.payload = payload
        self.position = pos
        self.color = np.random.rand(3)


class HitDetector:
    """
    Wrapper class for commonly used hit detectors in the voxelization process.
    """

    def __init__(self, detector):
        self.detector = detector

    def __call__(self, position, size):
        return self.detector(position, size)

    @classmethod
    def for_rtree(cls, tree):
        """
        Factory function that creates a hit detector for the given morphology.

        :param morphology: A morphology.
        :type morphology: :class:`TrueMorphology`
        :returns: A hit detector
        :rtype: :class:`HitDetector`
        """
        # Create the detector function
        def tree_detector(box_origin, box_size):
            # Report a hit if more than 0 compartments are within the box.
            return len(detect_box_compartments(tree, box_origin, box_size)) > 0

        # Return the tree detector function as the factory product
        return cls(tree_detector)
