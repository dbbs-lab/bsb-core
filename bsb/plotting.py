import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .networks import all_depth_first_branches, get_branch_points, reduce_branch
import numpy as np, math, functools
from .morphologies import Compartment
from contextlib import contextmanager
import random


class CellTrace:
    def __init__(self, meta, data):
        self.meta = meta
        self.data = data


class CellTraces:
    def __init__(self, id, title, order=None):
        self.traces = []
        self.cell_id = id
        self.title = title
        self.order = order

    def add(self, meta, data):
        self.traces.append(CellTrace(meta, data))

    def __iter__(self):
        return iter(self.traces)

    def __len__(self):
        return len(self.traces)


class CellTraceCollection:
    def __init__(self, cells=None):
        if cells is None:
            cells = {}
        elif isinstance(cells, list):
            cells = dict(map(lambda cell: (cell.cell_id, cell), cells))
        self.cells = cells
        self.legends = []
        self.colors = []

    def set_legends(self, legends):
        self.legends = legends

    def set_colors(self, colors):
        self.colors = colors

    def add(self, id, meta, data):
        if not id in self.cells:
            self.cells[id] = CellTraces(
                id,
                meta.get("display_label", "Cell " + str(id)),
                order=meta.get("order", None),
            )
        self.cells[id].add(meta, data)

    def __iter__(self):
        return iter(self.cells.values())

    def __len__(self):
        return len(self.cells)

    def order(self):
        self.cells = dict(sorted(self.cells.items(), key=lambda t: t[1].order or 0))


def _figure(f):
    """
    Decorator for functions that produce a Figure. Can set defaults, create and show
    figures and disable the legend.

    Adds the `show` and `legend` keyword arguments.
    """

    @functools.wraps(f)
    def wrapper_function(*args, fig=None, show=True, legend=True, **kwargs):
        if fig is None:
            fig = go.Figure()
        r = f(*args, fig=fig, show=show, legend=legend, **kwargs)
        fig.update_layout(showlegend=legend)
        if show:
            fig.show()
        return r

    return wrapper_function


def _network_figure(f):
    """
    Decorator for functions that produce a Figure of a network. Applies ``@_figure``
    and can create cubic perspective and swap the Y & Z axis labels.

    Adds the `cubic` and `swapaxes` keyword arguments.
    """

    @functools.wraps(f)
    @_figure
    def wrapper_function(*args, fig=None, cubic=True, swapaxes=True, **kwargs):
        r = f(*args, fig=fig, cubic=cubic, swapaxes=swapaxes, **kwargs)
        if cubic:
            fig.update_layout(scene_aspectmode="cube")
        if swapaxes:
            axis_labels = dict(xaxis_title="X", yaxis_title="Z", zaxis_title="Y")
        else:
            axis_labels = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
        fig.update_layout(scene=axis_labels)
        return r

    return wrapper_function


def _input_highlight(f, required=False):
    """
    Decorator for functions that highlight an input region on a Figure.

    Adds the `input_region` keyword argument. Decorated function has to have a `fig`
    keyword argument.

    :param required: If set to True, an ArgumentError is thrown if no `input_region`
      is specified
    :type required: bool
    """

    @functools.wraps(f)
    def wrapper_function(*args, fig=None, input_region=None, **kwargs):
        r = f(*args, fig=fig, **kwargs)
        if input_region is not None:
            shapes = [
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=input_region[0],
                    y0=0,
                    x1=input_region[1],
                    y1=1,
                    fillcolor="#d3d3d3",
                    opacity=0.3,
                    line=dict(width=0),
                )
            ]
            fig.update_layout(shapes=shapes)
        elif required:
            raise ArgumentError("Required keyword argument `input_region` omitted.")
        return r

    return wrapper_function


def _plot_network(network, fig, swapaxes):
    for type in network.configuration.cell_types.values():
        if type.entity:
            continue
        pos = network.cells_by_type[type.name][:, [2, 3, 4]]
        color = type.plotting.color
        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1 if not swapaxes else 2],
                z=pos[:, 2 if not swapaxes else 1],
                mode="markers",
                marker=dict(color=color, size=type.placement.radius),
                opacity=type.plotting.opacity,
                name=type.plotting.label,
            )
        )


@_network_figure
def plot_network(
    network, fig=None, cubic=True, swapaxes=True, show=True, legend=True, from_memory=True
):
    """
    Plot a network, either from the current cache or the storage.
    """
    if from_memory:
        _plot_network(network, fig, swapaxes)
    else:
        network.reset_network_cache()
        for type in network.configuration.cell_types.values():
            if type.entity:
                continue
            # Load from HDF5
            network.get_cells_by_type(type.name)
        _plot_network(network, fig, swapaxes)
    return fig


@_network_figure
def plot_detailed_network(
    network, fig=None, cubic=True, swapaxes=True, show=True, legend=True, ids=None
):
    from .output import MorphologyRepository

    ms = MorphologyScene(fig)
    mr = network.morphology_repository

    for cell_type in network.configuration.cell_types.values():
        segment_radius = 1.0
        if cell_type.name != "granule_cell":
            segment_radius = 2.5
        m_names = cell_type.list_all_morphologies()
        if len(m_names) == 0:
            continue
        if len(m_names) > 1:
            raise NotImplementedError(
                "We haven't implemented plotting different morphologies per cell type yet. Open an issue if you need it."
            )
        cells = network.get_placement_set(cell_type.name).cells
        morpho = mr.get_morphology(m_names[0])
        for cell in cells:
            if ids is not None and cell.id not in ids:
                continue
            ms.add_morphology(
                morpho,
                cell.position,
                color=cell_type.plotting.color,
                soma_radius=cell_type.placement.soma_radius,
                segment_radius=segment_radius,
            )
    ms.prepare_plot()
    scene = fig.layout.scene
    scene.xaxis.range = [-200, 200]
    scene.yaxis.range = [-200, 200]
    scene.zaxis.range = [0, 600]
    return fig


def get_voxel_cloud_traces(
    cloud, selected_voxels=None, offset=[0.0, 0.0, 0.0], color=None
):
    # Calculate the 3D voxel indices based on the voxel positions and the grid size.
    boxes = cloud.get_boxes()
    voxels = cloud.voxels.copy()
    box_positions = np.column_stack(boxes[:, voxels])
    # Calculate normalized occupancy of each voxel to determine transparency
    occupancies = cloud.get_occupancies() / 1.5
    if color is None:
        color = [255, 0, 0]
    color = list(map(str, color))

    if color is None:
        color = [0.0, 255.0, 0.0]
    color = [str(c) for c in color]

    colors = np.empty(voxels.shape, dtype=object)
    if selected_voxels is not None:
        # Color selected voxels
        colors[voxels] = "rgba(0, 0, 0, 0.0)"
        colors[selected_voxels] = "rgba(" + ",".join(color) + ", 1.0)"
    else:
        # Prepare voxels with the compartment density coded into the alpha of the facecolor
        colors[voxels] = [
            "rgba(" + ",".join(color) + ", {})".format(o) for o in occupancies
        ]
    traces = []
    for box, color in zip(box_positions, colors[voxels]):
        box += offset
        traces.extend(
            plotly_block(box, [cloud.grid_size, cloud.grid_size, cloud.grid_size], color)
        )

    return traces


@_network_figure
def plot_voxel_cloud(
    cloud,
    selected_voxels=None,
    fig=None,
    show=True,
    legend=True,
    cubic=True,
    swapaxes=True,
    set_range=True,
    color=None,
    offset=[0.0, 0.0, 0.0],
):
    traces = get_voxel_cloud_traces(
        cloud, selected_voxels=selected_voxels, offset=offset, color=color
    )
    for trace in traces:
        fig.add_trace(trace)
    if set_range:
        box = cloud.get_voxel_box()
        range = [min(box), max(box)]
        fig.layout.scene.xaxis.range = range + offset[0]
        fig.layout.scene.yaxis.range = range + offset[2] if swapaxes else offset[1]
        fig.layout.scene.zaxis.range = range + offset[1] if swapaxes else offset[2]
    return fig


def get_branch_trace(compartments, offset=[0.0, 0.0, 0.0], color="black", width=1.0):
    x = [c.start[0] + offset[0] for c in compartments]
    y = [c.start[1] + offset[1] for c in compartments]
    z = [c.start[2] + offset[2] for c in compartments]
    # Add branch endpoint
    x.append(compartments[-1].end[0] + offset[0])
    y.append(compartments[-1].end[1] + offset[1])
    z.append(compartments[-1].end[2] + offset[2])
    return go.Scatter3d(
        x=x, y=z, z=y, mode="lines", line=dict(width=width, color=color), showlegend=False
    )


def get_soma_trace(
    soma_radius, offset=[0.0, 0.0, 0.0], color="black", opacity=1, steps=5
):
    phi = np.linspace(0, 2 * np.pi, num=steps * 2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=steps)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi) * soma_radius + offset[0]
    y = np.cos(theta) * np.cos(phi) * soma_radius + offset[2]
    z = np.sin(theta) * soma_radius + offset[1]

    return go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        opacity=opacity,
        color=color,
        alphahull=0,
    )


@_network_figure
def plot_fiber_morphology(
    fiber,
    offset=[0.0, 0.0, 0.0],
    fig=None,
    cubic=True,
    swapaxes=True,
    show=True,
    legend=True,
    set_range=True,
    color="black",
    segment_radius=1.0,
):
    def get_branch_traces(branches, traces):
        for branch in branches:
            traces.append(
                get_branch_trace(
                    branch._compartments, offset, color=color, width=segment_radius
                )
            )
            get_branch_traces(branch.child_branches, traces)

    traces = []
    get_branch_traces(fiber.root_branches, traces)

    for trace in traces:
        fig.add_trace(trace)
    return fig


@_network_figure
def plot_morphology(
    morphology,
    offset=[0.0, 0.0, 0.0],
    fig=None,
    cubic=True,
    swapaxes=True,
    show=True,
    legend=True,
    set_range=True,
    color="black",
    reduce_branches=False,
    soma_radius=None,
    segment_radius=1.0,
):
    compartments = np.array(morphology.compartments.copy())
    dfs_list = all_depth_first_branches(morphology.get_compartment_network())
    if reduce_branches:
        branch_points = get_branch_points(dfs_list)
        dfs_list = list(map(lambda b: reduce_branch(b, branch_points), dfs_list))
    traces = []
    for branch in dfs_list[::-1]:
        branch_comps = compartments[branch]
        width = _get_branch_width(branch_comps, segment_radius)
        traces.append(get_branch_trace(branch_comps, offset, color=color, width=width))
    traces.append(
        get_soma_trace(
            soma_radius if soma_radius is not None else compartments[0].radius,
            offset,
            color,
        )
    )
    for trace in traces:
        fig.add_trace(trace)
    if set_range:
        set_scene_range(fig.layout.scene, morphology.get_plot_range(offset=offset))
    return fig


@_figure
def plot_intersections(
    from_morphology,
    from_pos,
    to_morphology,
    to_pos,
    intersections,
    offset=[0.0, 0.0, 0.0],
    fig=None,
    show=True,
    legend=True,
):
    from_compartments = (
        np.array(from_morphology.compartment_tree.get_arrays()[0])
        + np.array(offset)
        + np.array(from_pos)
    )
    to_compartments = (
        np.array(to_morphology.compartment_tree.get_arrays()[0])
        + np.array(offset)
        + np.array(to_pos)
    )


def _get_branch_width(branch, seg_radius):
    width = seg_radius
    try:
        if isinstance(seg_radius, dict):
            branch_type = branch[-1].type
            bt = int(branch_type / 100) if branch_type > 100 else int(branch_type)
            width = seg_radius[bt]
    except KeyError:
        raise Exception("Plotting width not specified for branches of type " + str(bt))
    return width


def plot_block(fig, origin, sizes, color=None, colorscale="Cividis", **kwargs):
    edges, faces = plotly_block(origin, sizes, color, colorscale)
    # fig.add_trace(edges, **kwargs)
    fig.add_trace(faces, **kwargs)


def plotly_block(origin, sizes, color=None, colorscale_value=None, colorscale="Cividis"):
    return plotly_block_edges(origin, sizes), plotly_block_faces(origin, sizes, color)


def plotly_block_faces(
    origin,
    sizes,
    color=None,
    colorscale_value=None,
    colorscale="Cividis",
    cmin=0,
    cmax=16.0,
):
    # 8 vertices of a block
    x = origin[0] + np.array([0, 0, 1, 1, 0, 0, 1, 1]) * sizes[0]
    y = origin[1] + np.array([0, 1, 1, 0, 0, 1, 1, 0]) * sizes[1]
    z = origin[2] + np.array([0, 0, 0, 0, 1, 1, 1, 1]) * sizes[2]
    color_args = {}
    if colorscale_value:
        color_args = {
            "colorscale": colorscale,
            "intensity": np.ones((8)) * colorscale_value,
            "cmin": cmin,
            "cmax": cmax,
        }
    if color:
        color_args = {"color": color}
    return go.Mesh3d(
        x=x,
        y=z,
        z=y,
        # i, j and k give the vertices of the mesh triangles
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.3,
        **color_args
    )


def plotly_block_edges(origin, sizes):
    x = origin[0] + np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) * sizes[0]
    y = origin[1] + np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]) * sizes[1]
    z = origin[2] + np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]) * sizes[2]
    return go.Scatter3d(
        x=x, y=z, z=y, mode="lines", line=dict(width=1.0, color="black"), showlegend=False
    )


def set_scene_range(scene, bounds):
    if hasattr(scene, "layout"):
        scene = scene.layout.scene  # Scene was a figure
    scene.xaxis.range = bounds[0]
    scene.yaxis.range = bounds[2]
    scene.zaxis.range = bounds[1]


def set_morphology_scene_range(scene, offset_morphologies):
    """
    Set the range on a scene containing multiple morphologies.

    :param scene: A scene of the figure. If the figure itself is given, ``figure.layout.scene`` will be used.
    :param offset_morphologies: A list of tuples where the first element is offset and the 2nd is the :class:`Morphology`
    """
    bounds = np.array(list(map(lambda m: m[1].get_plot_range(m[0]), offset_morphologies)))
    combined_bounds = np.array(
        list(zip(np.min(bounds, axis=0)[:, 0], np.max(bounds, axis=0)[:, 1]))
    )
    span = max(map(lambda b: b[1] - b[0], combined_bounds))
    combined_bounds[:, 1] = combined_bounds[:, 0] + span
    set_scene_range(scene, combined_bounds)


def hdf5_plot_spike_raster(spike_recorders, input_region=None, show=True):
    """
    Create a spike raster plot from an HDF5 group of spike recorders.
    """
    x = {}
    y = {}
    colors = {}
    ids = {}
    for cell_id, dataset in spike_recorders.items():
        attrs = dict(dataset.attrs)
        if len(dataset.shape) == 1 or dataset.shape[1] == 1:
            times = dataset[()]
            set_ids = np.ones(len(times)) * int(
                attrs.get("cell_id", attrs.get("cell", cell_id))
            )
        else:
            times = dataset[:, 1]
            set_ids = dataset[:, 0]
        label = attrs.get("label", "unlabelled")
        if not label in x:
            x[label] = []
        if not label in y:
            y[label] = []
        if not label in colors:
            colors[label] = attrs.get("color", "black")
        if not label in ids:
            ids[label] = 0
        ids[label] += 1
        # Add the spike timings on the X axis.
        x[label].extend(times)
        # Set the cell id for the Y axis of each added spike timing.
        y[label].extend(set_ids)
    # Use the parallel arrays x & y to plot a spike raster
    fig = go.Figure(
        layout=dict(
            xaxis=dict(title_text="Time (ms)"), yaxis=dict(title_text="Cell (ID)")
        )
    )
    sort_by_size = lambda d: {k: v for k, v in sorted(d.items(), key=lambda i: len(i[1]))}
    start_id = 0
    for label, x, y in [(label, x[label], y[label]) for label in sort_by_size(x).keys()]:
        y = [yi + start_id for yi in y]
        start_id += ids[label]
        plot_spike_raster(
            x,
            y,
            label=label,
            fig=fig,
            show=False,
            color=colors[label],
            input_region=input_region,
        )
    if show:
        fig.show()
    return fig


def hdf5_gdf_plot_spike_raster(spike_recorders, input_region=None, fig=None, show=True):
    """
    Create a spike raster plot from an HDF5 group of spike recorders saved from NEST gdf files.
    Each HDF5 dataset includes the spike timings of the recorded cell populations, with spike
    times in the first row and neuron IDs in the second row.
    """

    cell_ids = [np.unique(spike_recorders[k][:, 1]) for k in spike_recorders.keys()]
    x = {}
    y = {}
    colors = {}
    ids = {}

    for cell_id, dataset in spike_recorders.items():
        data = dataset[:, 0]
        neurons = dataset[:, 1]
        attrs = dict(dataset.attrs)
        label = attrs["label"]
        colors[label] = attrs["color"]
        if not label in x:
            x[label] = []
        if not label in y:
            y[label] = []
        if not label in colors:
            colors[label] = attrs["color"]
        if not label in ids:
            ids[label] = 0
        cell_id = ids[label]
        ids[label] += 1
        # Add the spike timings on the X axis.
        x[label].extend(data)
        # Set the cell id for the Y axis of each added spike timing.
        y[label].extend(neurons)

    subplots_fig = make_subplots(cols=1, rows=len(x), subplot_titles=list(x.keys()))
    _min = float("inf")
    _max = -float("inf")
    for i, (c, t) in enumerate(x.items()):
        _min = min(_min, np.min(np.array(t)))
        _max = max(_max, np.max(np.array(t)))
    subplots_fig.update_xaxes(range=[_min, _max])
    # Overwrite the layout and grid of the single plot that is handed to us
    # to turn it into a subplots figure.
    fig._grid_ref = subplots_fig._grid_ref
    fig._layout = subplots_fig._layout
    for i, l in enumerate((x.keys())):
        plot_spike_raster(
            x[l],
            y[l],
            label=l,
            fig=fig,
            row=i + 1,
            col=1,
            show=False,
            color=colors[l],
            input_region=input_region,
            **kwargs
        )
    if show:
        fig.show()
    return fig


@_figure
@_input_highlight
def plot_spike_raster(
    spike_timings,
    cell_ids,
    fig=None,
    row=None,
    col=None,
    show=True,
    legend=True,
    label="Cells",
    color=None,
):
    fig.add_trace(
        go.Scatter(
            x=spike_timings,
            y=cell_ids,
            mode="markers",
            marker=dict(symbol="square", size=2, color=color or "black"),
            name=label,
        ),
        row=row,
        col=col,
    )


def hdf5_gather_voltage_traces(handle, root, groups=None):
    if not groups:
        groups = [""]
    traces = CellTraceCollection()
    for group in groups:
        path = root + group
        for name, dataset in handle[path].items():
            meta = {}
            id = int(name.split(".")[0])
            meta["id"] = id
            meta["location"] = name
            meta["group"] = path
            for k, v in dataset.attrs.items():
                meta[k] = v
            traces.add(id, meta, dataset)
    return traces


@_figure
@_input_highlight
def plot_traces(traces, fig=None, show=True, legend=True):
    traces.order()
    subplots_fig = make_subplots(
        cols=1, rows=len(traces), subplot_titles=[trace.title for trace in traces]
    )
    subplots_fig.update_layout(height=len(traces) * 130)
    # Overwrite the layout and grid of the single plot that is handed to us
    # to turn it into a subplots figure.
    fig._grid_ref = subplots_fig._grid_ref
    fig._layout = subplots_fig._layout
    legend_groups = set()
    legends = traces.legends
    for i, cell_traces in enumerate(traces):
        for j, trace in enumerate(cell_traces):
            showlegend = legends[j] not in legend_groups
            fig.append_trace(
                go.Scatter(
                    x=trace.data[:, 0],
                    y=trace.data[:, 1],
                    legendgroup=legends[j],
                    name=legends[j],
                    showlegend=showlegend,
                    mode="lines",
                    marker=dict(color=traces.colors[j]),
                ),
                col=1,
                row=i + 1,
            )
            legend_groups.add(legends[j])
    return fig


class PSTH:
    def __init__(self):
        self.rows = []

    def add_row(self, row):
        row.index = len(self.rows)
        self.rows.append(row)

    def ordered_rows(self):
        return sorted(self.rows, key=lambda t: t.order or 0)


class PSTHStack:
    def __init__(self, name, color):
        self.name = name
        self.color = str(color)
        self.cells = 0
        self._included_ids = {0: np.empty(0)}
        self.list = []

    def extend(self, arr, run=0):
        self.list.extend(arr[:, 1])
        if run not in self._included_ids:
            self._included_ids[run] = np.empty(0)
        # Count all of the cells across the runs, but count unique cells per run
        self._included_ids[run] = np.unique(
            np.concatenate((self._included_ids[run], arr[:, 0]))
        )
        self.cells = sum(map(len, self._included_ids.values()))


class PSTHRow:
    def __init__(self, name, color, order=0):
        from colour import Color

        self.name = name
        color = Color(color) if color else Color(pick_for=random.random())
        self.palette = list(color.range_to("black", 6))
        self.stacks = {}
        self.max = -float("inf")
        self.order = order

    def extend(self, arr, stack=None, run=0):
        if stack not in self.stacks:
            self.stacks[stack] = PSTHStack(
                stack or self.name, self.palette[len(self.stacks)]
            )
        self.stacks[stack].extend(arr, run=run)
        self.max = max(self.max, np.max(arr[:, 1])) if len(arr) > 0 else self.max


@_figure
def hdf5_plot_psth(handle, duration=3, cutoff=0, start=0, fig=None, mod=None, **kwargs):
    psth = PSTH()
    row_map = {}
    for g in handle.values():
        l = g.attrs.get("label", "unlabelled")
        if l not in row_map:
            color = g.attrs.get("color", None)
            order = g.attrs.get("order", 0)
            row_map[l] = row = PSTHRow(l, color, order=order)
            psth.add_row(row)
        else:
            row = row_map[l]
        run_id = g.attrs.get("run_id", 0)
        adjusted = g[()]
        adjusted[:, 1] = adjusted[:, 1] - cutoff
        row.extend(adjusted, stack=g.attrs.get("stack", None), run=run_id)
    subplots_fig = make_subplots(
        cols=1,
        rows=len(psth.rows),
        subplot_titles=[row.name for row in psth.ordered_rows()],
        x_title=kwargs.get("x_title", "Time (ms)"),
        y_title=kwargs.get("y_title", "Population firing rate (Hz)"),
    )
    _max = -float("inf")
    for i, row in enumerate(psth.rows):
        _max = max(_max, row.max)
    subplots_fig.update_xaxes(range=[start, _max])
    subplots_fig.update_layout(title_text=kwargs.get("title", "PSTH"))
    # Allow the original figure to be updated before messing with it.
    if mod is not None:
        mod(subplots_fig)
    # Overwrite the layout and grid of the single plot that is handed to us
    # to turn it into a subplots figure. All modifications except for adding traces
    # should happen before this point.
    fig._grid_ref = subplots_fig._grid_ref
    fig._layout = subplots_fig._layout
    for i, row in enumerate(psth.ordered_rows()):
        for name, stack in sorted(row.stacks.items(), key=lambda x: x[0]):
            counts, bins = np.histogram(stack.list, bins=np.arange(start, _max, duration))
            if str(name).startswith("##"):
                # Lazy way to order the stacks; Stack names can start with ## and a number
                # and it will be sorted by name, but the ## and number are not displayed.
                name = name[4:]
            trace = go.Bar(
                x=bins,
                y=counts / stack.cells * 1000 / duration,
                name=name or row.name,
                marker=dict(color=stack.color),
            )
            fig.add_trace(trace, row=i + 1, col=1)
    return fig


class MorphologyScene:
    def __init__(self, fig=None):
        self.fig = fig or go.Figure()
        self._morphologies = []

    def add_morphology(self, morphology, offset=[0.0, 0.0, 0.0], **kwargs):
        self._morphologies.append((offset, morphology, kwargs))

    def show(self):
        self.prepare_plot()
        self.fig.show()

    def prepare_plot(self):
        if len(self._morphologies) == 0:
            raise MorphologyError("Cannot show empty MorphologyScene")
        for o, m, k in self._morphologies:
            plot_morphology(m, offset=o, show=False, fig=self.fig, **k)
        set_morphology_scene_range(self.fig.layout.scene, self._morphologies)
