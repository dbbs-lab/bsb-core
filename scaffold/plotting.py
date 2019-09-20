import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plotNetwork(scaffold, file=None, from_memory=False, block=True):
    if from_memory:
        plt.interactive(True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for type in scaffold.configuration.cell_types.values():
            pos = scaffold.cells_by_type[type.name][:, [2,3,4]]
            color = type.plotting.color
            ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],c=color)
        plt.show(block=block)

def plot_voxel_cloud(cloud):
    indices = np.array(cloud.positions / cloud.grid_size, dtype=int)
    indices[:, 0] -= min(indices[:, 0])
    indices[:, 1] -= min(indices[:, 1])
    indices[:, 2] -= min(indices[:, 2])
    x_max = max(indices[:, 0])
    y_max = max(indices[:, 1])
    z_max = max(indices[:, 2])
    maxmax = max(x_max, y_max, z_max) + 1
    grid_dimensions = (x_max + 1, y_max + 1, z_max + 1)
    voxel_occupancy = np.array(list(map(lambda x: len(x), cloud.map)))
    max_voxel_occupancy = max(voxel_occupancy)
    normalized_voxel_occupancy = voxel_occupancy / (max_voxel_occupancy * 1.5)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    voxels = np.zeros(grid_dimensions)
    colors = np.empty(voxels.shape, dtype=object)
    for i in range(indices.shape[0]):
        voxels[indices[i,0],indices[i,1],indices[i,2],] = True
        colors[indices[i,0],indices[i,1],indices[i,2],] = (1., 0., 0., normalized_voxel_occupancy[i])

    ax.set(xlim=(0., maxmax), ylim=(0., maxmax), zlim=(0., maxmax))
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=.25)
    plt.show(block=True)


# # The positions matrix, 3 columns: x, y, z
# positions = np.zeros((4, 3))
#
# positions[0, :] = [-2, 0, 0]
# positions[1, :] = [2, 2, 2]
# positions[2, :] = [2, -6, 2]
# positions[3, :] = [6, 8, 8]
#
# # The density map: a list of lists where each row is a voxel and the list of that row are all the compartments in that voxel
# _map = [
#     [4,5, 6, 8],
#     [4, 5],
#     [1],
#     []
# ]
#
# grid_size = 2
#
# voxel_cloud = type('VoxelCloud', (object,), {'positions': positions, 'map': _map, 'grid_size': grid_size})()
#
# plot_voxel_cloud(voxel_cloud)
