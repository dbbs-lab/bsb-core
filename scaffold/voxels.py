from scaffold.helpers import dimensions, origin
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from time import sleep

class VoxelCloud:
    def __init__(self, grid_size, positions, map):
        self.grid_size = grid_size
        self.positions = positions
        self.map = map

def voxelize(N, box_data, hit_detector, max_iterations=200, post_iterations=30):
    bounds = np.column_stack((box_data.origin - box_data.dimensions / 2, box_data.origin + box_data.dimensions / 2))
    last_box_lengths = np.zeros(len(box_data.dimensions))
    box_lengths = box_data.dimensions
    crossed_treshold = False
    searching = True
    post_i = 0
    i = 0
    last_box_count = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    plt.interactive(True)
    plt.show()
    artists = {}
    best_error = N
    best_lengths = box_data.dimensions.copy()
    while i < max_iterations and post_i < post_iterations:
        i += 1
        if crossed_treshold:
            post_i += 1
        box_count = 0
        boxes_x, boxes_y, boxes_z = np.mgrid[
            bounds[0, 0]:bounds[0, 1]:box_lengths[0],
            bounds[1, 0]:bounds[1, 1]:box_lengths[1],
            bounds[2, 0]:bounds[2, 1]:box_lengths[2]
        ]
        print('iteration',i)
        voxels = np.zeros((boxes_x.shape[0], boxes_y.shape[0], boxes_z.shape[0]), dtype=bool)
        plot_voxels = np.zeros((boxes_x.shape[0], boxes_z.shape[0], boxes_y.shape[0]), dtype=bool)
        for x_i in range(boxes_x.shape[0]):
            for y_i in range(boxes_x.shape[1]):
                for z_i in range(boxes_x.shape[1]):
                    x = boxes_x[x_i,y_i,z_i]
                    y = boxes_y[x_i,y_i,z_i]
                    z = boxes_z[x_i,y_i,z_i]
                    hit = hit_detector(np.array([x, y, z]), box_lengths)
                    voxels[x_i, y_i, z_i] = hit
                    plot_voxels[x_i, z_i, y_i] = hit
                    box_count += int(hit)

        ax.set(xlim=(0., boxes_x.shape[0]), ylim=(0., boxes_x.shape[1]), zlim=(0., boxes_x.shape[2]))
        for artist in artists.values():
            artist.remove()
        artists = ax.voxels(plot_voxels, facecolors=(1.,0.,0.,0.2), edgecolor='k', linewidth=.25)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # sleep(.5)
        if last_box_count < N and box_count >= N: # We're in the right range, now execute post_iterations to refine
            crossed_treshold = True
        if box_count < N: # Decrease the box length
            new_box_lengths = box_lengths - np.abs(box_lengths - last_box_lengths) / 2
        else:
            new_box_lengths = box_lengths + np.abs(box_lengths - last_box_lengths) / 2
        last_box_lengths = box_lengths
        box_lengths = new_box_lengths
        last_box_count = box_count
        print('lengths:', box_lengths)
        print('hits:', box_count)
        if abs(N - box_count) <= best_error:
            print('new best hit!!!!! error:', abs(N - box_count))
            best_error = abs(N - box_count)
            best_lengths = last_box_lengths
            best_boxes = np.array([boxes_x, boxes_y, boxes_z])
            best_voxels = voxels
            best_plot_voxels = plot_voxels
    plt.close()
    return best_boxes, best_voxels, best_lengths, best_error, best_plot_voxels

class BoxData(dimensions, origin):
    pass
