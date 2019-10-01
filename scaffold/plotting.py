import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .networks import depth_first_branches
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
    occupancies = cloud.get_occupancies() / 1.5

    colors = np.empty(voxels.shape, dtype=object)
    if not selected_voxels is None:
        # Don't double draw the selected voxels (might be more performant and insignificant to just double draw)
        voxels[selected_voxels] = False
        # Color selected voxels
        colors[voxels] = (0., 0., 0., 0.)
        colors[selected_voxels] = (0., 1., 0., 1.)
    else:
        # Prepare voxels with the compartment density coded into the alpha of the facecolor
        colors[voxels] = list(map(lambda o: (1., 0., 0., o), occupancies))
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

def get_branch_trace(compartments):
    x = [c.start[0] for c in compartments]
    y = [c.start[1] for c in compartments]
    z = [c.start[2] for c in compartments]
    # Add branch endpoint
    x.append(compartments[-1].end[0])
    y.append(compartments[-1].end[1])
    z.append(compartments[-1].end[2])
    return go.Scatter3d(
        x=x, y=z, z=y, mode='lines',
        line=dict(
            width=1.,
            color=(0., 0., 0., 1.)
        )
    )

def plot_morphology(morphology, return_traces=False, compartment_selection=()):
    compartments = morphology.compartments.copy()
    compartments.insert(0, type('Compartment', (object,), {'start': [0., 0., 0.], 'end': [0., 0., 0.]})())
    compartments = np.array(compartments)
    dfs_list = depth_first_branches(morphology.get_compartment_network())
    traces = []
    c = 0
    for branch in dfs_list[::-1]:
        c += 1
        print('busy...', c)
        traces.append(get_branch_trace(compartments[branch]))
    print('traces made')
    if compartment_selection != (): # A selection is being made
        # Get selected compartments
        highlighted = compartments[compartment_selection]
    if return_traces:
        return traces
    else:
        fig = go.Figure(data=traces)
        fig.update_layout(showlegend=False)
        set_3D_axes_range(fig, morphology.get_plot_range())
        fig.show()
        print('shown')

def set_3D_axes_range(fig, bounds, row=None, **kwargs):
    print(bounds)
    fig.update_layout(scene_xaxis_range=bounds[0],scene_yaxis_range=bounds[2],scene_zaxis_range=bounds[1], **kwargs)

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

def plot_voxelize_results(bounds, voxels, box_length, color=(1.,0.,0.,0.2)):
    plot_voxels = np.swapaxes(voxels, 1, 2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set(xlabel='x', ylabel='z', zlabel='y')
    maxmax = np.max(voxels.shape)
    ax.set(xlim=(0., maxmax), ylim=(0., maxmax), zlim=(0., maxmax))
    ax.voxels(plot_voxels, facecolors=color, edgecolor='k', linewidth=.25)
    # plt.show(block=True)
    return fig, ax

def plot_eli_voxels(morphology, voxel_positions, voxel_compartment_map):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    c = 0
    for trace in plot_morphology(morphology, return_traces=True):
        c += 1
        print('adding... ', c)
        fig.add_trace(
            trace,
            row=1, col=1
        )
    fig.update_layout(showlegend=False)
    set_3D_axes_range(fig, morphology.get_plot_range(),row=1,col=1)
    print('writing...')
    fig.write_html("test_figure.html", auto_open=True)
    print('written')
