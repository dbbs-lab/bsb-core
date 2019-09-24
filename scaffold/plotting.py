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

def plot_voxel_cloud(cloud, fig_ax_tuple=None, selected_voxel_id=None):
    # Calculate the 3D voxel indices based on the voxel positions and the grid size.
    indices = np.array(cloud.positions / cloud.grid_size, dtype=int)
    # Translate the voxel cloud to 0, 0, 0 as minimum index.
    min_x = min(indices[:, 0])
    min_y = min(indices[:, 1])
    min_z = min(indices[:, 2])
    if not selected_voxel_id is None:
        selected_voxel = np.array(cloud.positions[selected_voxel_id] / cloud.grid_size, dtype=int)
        selected_voxel[0] -= min_x
        selected_voxel[1] -= min_y
        selected_voxel[2] -= min_z
    indices[:, 0] -= min_x
    indices[:, 1] -= min_y
    indices[:, 2] -= min_z
    # Determine the total grid dimensions
    x_max = max(indices[:, 0])
    y_max = max(indices[:, 1])
    z_max = max(indices[:, 2])
    maxmax = max(x_max, y_max, z_max) + 1
    grid_dimensions = (x_max + 1, y_max + 1, z_max + 1)
    # Calculate normalized occupancy of each voxel to determine transparency
    voxel_occupancy = np.array(list(map(lambda x: len(x), cloud.map)))
    max_voxel_occupancy = max(voxel_occupancy)
    normalized_voxel_occupancy = voxel_occupancy / (max_voxel_occupancy * 1.5)
    voxels = np.zeros(grid_dimensions)
    colors = np.empty(voxels.shape, dtype=object)
    if not selected_voxel is None:
        # Switch on the voxel at the selected index positions, switch off everywhere else
        # Select another voxel that isn't the selected voxel, to overwrite the position of the selected
        # voxel's indices with, so that they both refer to the same unselected voxel.
        i = 0 if selected_voxel_id != 0 else 1
        indices[selected_voxel_id, 0] = indices[i, 0]
        indices[selected_voxel_id, 1] = indices[i, 1]
        indices[selected_voxel_id, 2] = indices[i, 2]
        # Color the selected voxel
        voxels[selected_voxel[0],selected_voxel[1],selected_voxel[2],] = True
        colors[selected_voxel[0],selected_voxel[1],selected_voxel[2],] = (0., 1., 0., 1.)
        # Create transparent unselected voxels
        for i in range(indices.shape[0]):
            voxels[indices[i,0],indices[i,1],indices[i,2],] = True
            colors[indices[i,0],indices[i,1],indices[i,2],] = (0., 0., 0., 0.)
    else:
        # Switch on the voxels on the selected index positions
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

def plot_compartment(ax, compartment, radius_multiplier=1., max_radius=4., color=None):
    artist = ax.plot_wireframe(
        np.array([compartment.start[0], compartment.end[0]]),
        np.array([compartment.start[1], compartment.end[1]]),
        np.array([[compartment.start[2], compartment.end[2]]]),
        linewidth=min(compartment.radius * radius_multiplier, max_radius),
        color=color
    )

def plot_morphology(morphology, fig_ax_tuple=None, compartment_selection=()):
    compartments = np.array(morphology.compartments)
    if fig_ax_tuple is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_ax_tuple
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    if compartment_selection != (): # A selection is being made
        # Get selected compartments
        highlighted = np.array([x.end for x in compartments[compartment_selection]])
        # Draw a faded out morphology
        for faded_compartment in compartments:
            plot_compartment(ax, faded_compartment, color=(0.3,0.3,0.3,0.6))
        # Mark the selected compartments
        ax.scatter(*highlighted.transpose(), s=5, c='red', marker="^")
    else: # No selection is being made
        # Style all compartments normally
        for compartment in compartments:
            plot_compartment(ax, compartment)


    ax.scatter(*compartments[1].end, s=compartments[1].radius ** 2, c='red')

def plot_voxel_morpho_map(morphology, cloud, selected_voxel_id=None, compartment_selection=()):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_cloud = fig.add_subplot(1, 2, 1, projection='3d')
    ax_frame = fig.add_subplot(1, 2, 2, projection='3d')
    plot_voxel_cloud(cloud, fig_ax_tuple=(fig, ax_cloud), selected_voxel_id=selected_voxel_id)
    plot_morphology(
        morphology,
        fig_ax_tuple=(fig, ax_frame),
        compartment_selection=compartment_selection
    )
    plt.show(block=True)

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
positions = np.zeros((4, 3))

positions[0, :] = [-2, 0, 0]
positions[1, :] = [2, 2, 2]
positions[2, :] = [2, 6, 2]
positions[3, :] = [6, 8, 8]

# The density map: a list of lists where each row is a voxel and the list of that row are all the compartments in that voxel
_map = [
    [4,5, 6, 8],
    [4, 5],
    [1],
    []
]

grid_size = 2

voxel_cloud = type('VoxelCloud', (object,), {'positions': positions, 'map': _map, 'grid_size': grid_size})()
# morphology = type('Morphology', (object,), {'compartments': compartments})()
#
# # plot_voxel_morpho_map(morphology, voxel_cloud)
# # plt.show(block=True)
