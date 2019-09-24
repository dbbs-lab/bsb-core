import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np, math

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

def plot_voxel_cloud(cloud, fig_ax_tuple=None):
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
    voxels = np.zeros(grid_dimensions)
    colors = np.empty(voxels.shape, dtype=object)
    for i in range(indices.shape[0]):
        voxels[indices[i,0],indices[i,1],indices[i,2],] = True
        colors[indices[i,0],indices[i,1],indices[i,2],] = (1., 0., 0., normalized_voxel_occupancy[i])

    if fig_ax_tuple is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_ax_tuple
    ax.set(xlim=(0., maxmax), ylim=(0., maxmax), zlim=(0., maxmax))
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=.25)

def plot_morphology(morphology, fig_ax_tuple=None, compartment_type=None):
    if fig_ax_tuple is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_ax_tuple
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    if compartment_type is None:
        compartments = morphology.compartments
    else:
        compartments = list(filter(lambda c: c.type == compartment_type, morphology.compartments))
    for compartment in compartments:
        artist = ax.plot_wireframe(
            np.array([compartment.start[0], compartment.end[0]]),
            np.array([compartment.start[1], compartment.end[1]]),
            np.array([[compartment.start[2], compartment.end[2]]]),
            linewidth=min(compartment.radius, 4)
        )
    ax.scatter(*morphology.compartments[1].end, s=morphology.compartments[1].radius ** 2, c='red')

def plot_voxel_morpho_map(morphology, cloud, morphology_compartment=None):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_cloud = fig.add_subplot(1, 2, 1, projection='3d')
    ax_frame = fig.add_subplot(1, 2, 2, projection='3d')
    plot_voxel_cloud(cloud, fig_ax_tuple=(fig, ax_cloud))
    plot_morphology(morphology, fig_ax_tuple=(fig, ax_frame), compartment_type=morphology_compartment)

# def line2d_seg_dist(p1, p2, p0):
#     """distance(s) from line defined by p1 - p2 to point(s) p0
#
#     p0[0] = x(s)
#     p0[1] = y(s)
#
#     intersection point p = p1 + u*(p2-p1)
#     and intersection point lies within segment if u is between 0 and 1
#     """
#
#     x21 = p2[0] - p1[0]
#     y21 = p2[1] - p1[1]
#     x01 = np.asarray(p0[0]) - p1[0]
#     y01 = np.asarray(p0[1]) - p1[1]
#
#     u = (x01*x21 + y01*y21) / (x21**2 + y21**2)
#     u = np.clip(u, 0, 1)
#     d = np.hypot(x01 - u*x21, y01 - u*y21)
#
#     return d
#
# def get_3d_pos(ax, xd, yd):
#     if ax.M is None:
#         return {}
#
#     p = (xd, yd)
#     edges = ax.tunit_edges()
#     ldists = [(line2d_seg_dist(p0, p1, p), i) for \
#                 i, (p0, p1) in enumerate(edges)]
#     ldists.sort()
#
#     # nearest edge
#     edgei = ldists[0][1]
#
#     p0, p1 = edges[edgei]
#
#     # scale the z value to match
#     x0, y0, z0 = p0
#     x1, y1, z1 = p1
#     d0 = np.hypot(x0-xd, y0-yd)
#     d1 = np.hypot(x1-xd, y1-yd)
#     dt = d0+d1
#     z = d1/dt * z0 + d0/dt * z1
#
#     x, y, z = mplot3d.proj3d.inv_transform(xd, yd, z, ax.M)
#     return type('pos_obj', (object,), {'x':x, 'y': y, 'z': z})()
#
# The positions matrix, 3 columns: x, y, z
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
# morphology = type('Morphology', (object,), {'compartments': compartments})()
#
# # plot_voxel_morpho_map(morphology, voxel_cloud)
# # plt.show(block=True)
