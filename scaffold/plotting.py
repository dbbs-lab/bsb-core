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

def plot_voxel_cloud(cloud, fig_ax_tuple=None, selected_voxels=None):
    # Calculate the 3D voxel indices based on the voxel positions and the grid size.
    boxes = cloud.get_boxes()
    voxels = cloud.voxels.copy()
    # Calculate normalized occupancy of each voxel to determine transparency
    voxel_occupancy = np.array(list(map(lambda x: len(x), cloud.map)))
    max_voxel_occupancy = max(voxel_occupancy)
    normalized_voxel_occupancy = list(map(lambda o: (1., 0., 0., o), voxel_occupancy / (max_voxel_occupancy * 1.5)))
    colors = np.empty(voxels.shape, dtype=object)
    if not selected_voxels is None:
        # Don't double draw the selected voxels (might be more performant and insignificant to just double draw)
        voxels[selected_voxels] = False
        # Color selected voxels
        colors[voxels] = (0., 0., 0., 0.)
        colors[selected_voxels] = (0., 1., 0., 1.)
    else:
        # Prepare voxels with the compartment density coded into the alpha of the facecolor
        colors[voxels] = normalized_voxel_occupancy
    # If no plotting tuple is provided, create a new figure
    if fig_ax_tuple is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_ax_tuple
    # Prepare plot
    ax.set(xlim=(0., voxels.shape[0]), ylim=(0., voxels.shape[0]), zlim=(0., voxels.shape[0]))
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    # Plot and return the voxel's artist dictionary
    return ax.voxels(np.swapaxes(voxels, 1, 2), facecolors=colors, edgecolor='k', linewidth=.25)

def plot_compartment(ax, compartment, radius_multiplier=1., max_radius=4., color=None):
    artist = ax.plot_wireframe(
        np.array([compartment.start[0], compartment.end[0]]),
        np.array([compartment.start[2], compartment.end[2]]),
        np.array([[compartment.start[1], compartment.end[1]]]),
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
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    if compartments.shape[0] > 1: # Just to be sure that we don't crash here on empty morphologies
        # Draw the cell soma.
        soma = compartments[1].end
        ax.scatter(soma[0], soma[2], soma[1], s=compartments[1].radius ** 2, c='blue')
    if compartment_selection != (): # A selection is being made
        # Get selected compartments
        highlighted = np.array([x.end for x in compartments[compartment_selection]])
        # Draw a faded out morphology
        for faded_compartment in compartments:
            plot_compartment(ax, faded_compartment, color=(0.3,0.3,0.3,0.6))
        # Mark the selected compartments
        t = highlighted.transpose()
        return ax.scatter(t[0], t[2], t[1], s=5, c='red', marker="^")
    else: # No selection is being made
        # Style all compartments normally
        for compartment in compartments:
            plot_compartment(ax, compartment)
        return None


def plot_voxel_morpho_map(morphology, selected_voxel_ids=None, compartment_selection=()):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_cloud = fig.add_subplot(1, 2, 1, projection='3d')
    ax_frame = fig.add_subplot(1, 2, 2, projection='3d')
    voxels = plot_voxel_cloud(morphology.cloud, fig_ax_tuple=(fig, ax_cloud), selected_voxel_ids=selected_voxel_ids)
    selection = plot_morphology(
        morphology,
        fig_ax_tuple=(fig, ax_frame),
        compartment_selection=compartment_selection
    )
    return fig, ax_cloud, ax_frame, voxels, selection

def plot_voxelize_results(bounds, voxels, box_length, error):
    plot_voxels = np.swapaxes(voxels, 1, 2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    maxmax = np.max(voxels.shape)
    ax.set(xlim=(0., maxmax), ylim=(0., maxmax), zlim=(0., maxmax))
    ax.voxels(plot_voxels, facecolors=(1.,0.,0.,0.2), edgecolor='k', linewidth=.25)
    plt.show(block=True)
