try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    pass
from .networks import depth_first_branches, get_branch_points, reduce_branch
import numpy as np, math
from .morphologies import Compartment
from contextlib import contextmanager

@contextmanager
def show_figure(fig=None, cubic=True, show=True, legend=True, swapaxes=True):
    try:
        if fig is None:
            fig = go.Figure()
        yield fig
    finally:
        fig.update_layout(showlegend=legend)
        if cubic:
            fig.update_layout(scene_aspectmode='cube')
        if swapaxes:
            fig.update_layout(scene = dict(
                                xaxis_title='X',
                                yaxis_title='Z',
                                zaxis_title='Y'))
        if show:
            fig.show()

def plot_network(scaffold, file=None, from_memory=True, block=True, show=True):
    if from_memory:
        with show_figure(show=show) as fig:
            for type in scaffold.configuration.cell_types.values():
                pos = scaffold.cells_by_type[type.name][:, [2,3,4]]
                color = type.plotting.color
                fig.add_trace(go.Scatter3d(
                    x=pos[:,0], y=pos[:,2], z=pos[:,1], mode='markers',
                    marker=dict(color=color, size=type.placement.radius),
                    name=type.plotting.display_name if hasattr(type.plotting, 'display_name') else type.name
                ))
            return fig
    else:
        raise NotImplementedError("Only network caches can be plot at this point.")

def get_voxel_cloud_traces(cloud, selected_voxels=None):
    # Calculate the 3D voxel indices based on the voxel positions and the grid size.
    boxes = cloud.get_boxes()
    voxels = cloud.voxels.copy()
    box_positions = np.column_stack(boxes[:, voxels])
    # Calculate normalized occupancy of each voxel to determine transparency
    occupancies = cloud.get_occupancies() / 1.5

    colors = np.empty(voxels.shape, dtype=object)
    if not selected_voxels is None:
        # Color selected voxels
        colors[voxels] = 'rgba(0, 0, 0, 0.0)'
        colors[selected_voxels] = 'rgba(0, 255, 0, 1.0)'
    else:
        # Prepare voxels with the compartment density coded into the alpha of the facecolor
        colors[voxels] = list(map(lambda o: 'rgba(255, 0, 0, {})'.format(o), occupancies))
    traces = []
    for box, color in zip(box_positions, colors[voxels]):
        traces.extend(plotly_block(box, [cloud.grid_size, cloud.grid_size, cloud.grid_size], color))

    return traces

def plot_voxel_cloud(cloud, bounds, selected_voxels=None):
    with show_figure(legend=False) as fig:
        traces = get_voxel_cloud_traces(cloud,selected_voxels=selected_voxels)
        for trace in traces:
            fig.add_trace(trace)
        set_scene_range(fig.layout.scene, bounds)

def get_branch_trace(compartments, offset = [0., 0., 0.], color='black', width=1.):
    x = [c.start[0] + offset[0] for c in compartments]
    y = [c.start[1] + offset[1] for c in compartments]
    z = [c.start[2] + offset[2] for c in compartments]
    # Add branch endpoint
    x.append(compartments[-1].end[0] + offset[0])
    y.append(compartments[-1].end[1] + offset[1])
    z.append(compartments[-1].end[2] + offset[2])
    return go.Scatter3d(
        x=x, y=z, z=y, mode='lines',
        line=dict(
            width=width,
            color=color
        )
    )

def get_soma_trace(soma_radius, offset=[0., 0., 0.], color='black'):
    theta = np.linspace(0,2*np.pi,10)
    phi = np.linspace(0,np.pi,10)
    x = np.outer(np.cos(theta),np.sin(phi)) * soma_radius + offset[0]
    y = np.outer(np.sin(theta),np.sin(phi)) * soma_radius + offset[2]
    z = np.outer(np.ones(10),np.cos(phi)) * soma_radius + offset[1]
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros(10),
        colorscale=[[0, color], [1, color]],
        showscale=False
    )

def plot_morphology(morphology, return_traces=False, offset=[0., 0., 0.], fig=None, show=True, set_range=True, color='black', reduce_branches=False, soma_radius=None, segment_radius=1.):
    compartments = morphology.compartments.copy()
    compartments.insert(0, Compartment([0, 0, *compartments[0].start, *compartments[0].end, 1., 0]))
    compartments = np.array(compartments)
    dfs_list = depth_first_branches(morphology.get_compartment_network())
    if reduce_branches:
        branch_points = get_branch_points(dfs_list)
        dfs_list = list(map(lambda b: reduce_branch(b, branch_points), dfs_list))
    traces = []
    for branch in dfs_list[::-1]:
        traces.append(get_branch_trace(compartments[branch], offset,color=color, width=segment_radius))
    traces.append(get_soma_trace(soma_radius if not soma_radius is None else compartments[0].radius, offset, color))
    if return_traces:
        return traces
    else:
        if fig is None:
            fig = go.Figure()
            fig.update_layout(showlegend=False)
        for trace in traces:
            fig.add_trace(trace)
        if set_range:
            set_scene_range(fig.layout.scene, morphology.get_plot_range(offset=offset))
        if show:
            fig.show()
        return fig

def plot_intersections(from_morphology, from_pos, to_morphology, to_pos, intersections, offset=[0., 0., 0.], fig=None):
    from_compartments = np.array(from_morphology.compartment_tree.get_arrays()[0]) + np.array(offset) + np.array(from_pos)
    to_compartments = np.array(to_morphology.compartment_tree.get_arrays()[0]) + np.array(offset) + np.array(to_pos)
    if fig is None:
        fig = go.Figure()
        fig.update_layout(showlegend=False)
        fig.show()

def plot_block(fig, origin, sizes, color=None, colorscale="Cividis", **kwargs):
    edges, faces = plotly_block(origin, sizes, color, colorscale)
    # fig.add_trace(edges, **kwargs)
    fig.add_trace(faces, **kwargs)

def plotly_block(origin, sizes, color=None, colorscale_value=None, colorscale="Cividis"):
    return plotly_block_edges(origin, sizes), plotly_block_faces(origin, sizes, color)

def plotly_block_faces(origin, sizes, color=None, colorscale_value=None, colorscale="Cividis",cmin=0,cmax=16.):
    # 8 vertices of a block
    x = origin[0] + np.array([0, 0, 1, 1, 0, 0, 1, 1]) * sizes[0]
    y = origin[1] + np.array([0, 1, 1, 0, 0, 1, 1, 0]) * sizes[1]
    z = origin[2] + np.array([0, 0, 0, 0, 1, 1, 1, 1]) * sizes[2]
    color_args = {}
    if colorscale_value:
        color_args = {'colorscale': colorscale, 'intensity': np.ones((8)) * colorscale_value, 'cmin': cmin, 'cmax': cmax}
    if color:
        color_args = {'color': color}
    return go.Mesh3d(
        x=x, y=z, z=y,
        # i, j and k give the vertices of the mesh triangles
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.3,
        **color_args
    )

def plotly_block_edges(origin, sizes):
    x = origin[0] + np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) * sizes[0]
    y = origin[1] + np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]) * sizes[1]
    z = origin[2] + np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]) * sizes[2]
    return go.Scatter3d(
        x=x, y=z, z=y, mode='lines',
        line=dict(
            width=1.,
            color='black'
        )
    )


def plot_eli_voxels(morphology, voxel_positions, voxel_compartment_map, selected_voxel_ids=None):
    if selected_voxel_ids is None:
        selected_voxel_ids = list(range(len(voxel_positions)))
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    )
    fig.update_layout(showlegend=False)
    with show_figure(fig=fig) as fig:
        for trace in plot_morphology(morphology, return_traces=True):
            fig.add_trace(
                trace,
                row=1, col=1
            )
        fig.update_layout(scene2_aspectmode='cube')
        # Determine voxel grid sizes.
        Δx, Δy, Δz = 0., 0., 0.
        no_dx, no_dy, no_dz = True, True, True
        for i in range(len(voxel_positions) - 1):
          if no_dx and voxel_positions[i, 0] != voxel_positions[i + 1, 0]:
            Δx = np.abs(voxel_positions[i, 0] - voxel_positions[i + 1, 0])
            no_dx = False
          if no_dy and voxel_positions[i, 1] != voxel_positions[i + 1, 1]:
            Δy = np.abs(voxel_positions[i, 1] - voxel_positions[i + 1, 1])
            no_dy = False
          if no_dz and voxel_positions[i, 2] != voxel_positions[i + 1, 2]:
            Δz = np.abs(voxel_positions[i, 2] - voxel_positions[i + 1, 2])
            no_dz = False
          if not no_dy and not no_dz and not no_dx:
            break
        # Resulting voxel grid sizes.
        Δ = [Δx, Δy, Δz]
        voxel_origins = np.min(voxel_positions, axis=0)
        total_grid_size = np.max(voxel_positions, axis=0) - voxel_origins
        diagonal = np.sum(total_grid_size ** 2)
        voxel_color_values = np.sum((voxel_positions - voxel_origins) ** 2, axis=1) / diagonal * 16.
        for voxel_id in range(len(voxel_color_values)):
            voxel = voxel_positions[voxel_id]
            voxel_compartments = voxel_compartment_map[voxel_id]
            if voxel_id in selected_voxel_ids:
                plot_block(fig, voxel, Δ, row=1, col=2, color=voxel_color_values[voxel_id] + 0.0001)
                fig.add_trace(
                    go.Scatter3d(
                        x = list(map(lambda c: morphology.compartments[c].end[0], voxel_compartments)),
                        y = list(map(lambda c: morphology.compartments[c].end[2], voxel_compartments)),
                        z = list(map(lambda c: morphology.compartments[c].end[1], voxel_compartments)),
                        mode='markers',
                        marker=dict(
                            size=2.,
                            cmin=0.,
                            cmax=16.,
                            colorscale_value=[voxel_color_values[voxel_id] for _ in range(len(voxel_compartments))],
                            colorscale='Viridis',
                        )
                    ), row=1, col=1
                )
            else:
                fig.add_trace(plotly_block_edges(voxel, Δ), row=1, col=2)
        fig.update_layout(scene_aspectmode='cube')
        fig.update_layout(scene2_aspectmode='cube')

def set_scene_range(scene, bounds):
    if hasattr(scene, "layout"):
        scene = scene.layout.scene # Scene was a figure
    scene.xaxis.range=bounds[0]
    scene.yaxis.range=bounds[2]
    scene.zaxis.range=bounds[1]

def set_morphology_scene_range(scene, offset_morphologies):
    bounds = np.array(list(map(lambda m: m[1].get_plot_range(m[0]), offset_morphologies)))
    combined_bounds = np.array(list(zip(np.min(bounds, axis=0)[:,0], np.max(bounds, axis=0)[:,1])))
    span = max(map(lambda b: b[1] - b[0], combined_bounds))
    combined_bounds[:,1] = combined_bounds[:,0] + span
    set_scene_range(scene, combined_bounds)
