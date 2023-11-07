import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np, math, functools
from contextlib import contextmanager
import random, types
from .reporting import warn
from colour import Color


class CellTrace:
    def __init__(self, meta, data):
        self.meta = meta
        self.data = data
        self.color = None


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

    def reorder(self, order):
        for o, key in zip(iter(order), self.cells.keys()):
            self.cells[key].order = o
        self.order()


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
    def wrapper_function(*args, fig=None, cubic=True, swapaxes=False, **kwargs):
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


def _morpho_figure(f):
    """
    Decorator for functions that produce a Figure of a morphology. Applies ``@_figure``
    and can set the offset, range & aspectratio and can swap the Y & Z axis labels.

    Adds the `offset`, `set_range` and `swapaxes` keyword arguments.
    """

    @functools.wraps(f)
    @_figure
    def wrapper_function(
        morphology,
        *args,
        offset=None,
        set_range=True,
        fig=None,
        swapaxes=False,
        soma_radius=None,
        **kwargs,
    ):
        if offset is None:
            offset = [0.0, 0.0, 0.0]
        r = f(
            morphology,
            *args,
            fig=fig,
            offset=offset,
            set_range=set_range,
            swapaxes=swapaxes,
            soma_radius=soma_radius,
            **kwargs,
        )
        if set_range and len(morphology):
            # Set the range to be the cube around the min and max of the
            # morphology
            rng = (
                np.array([np.min(morphology.bounds[0])] * 3),
                np.array([np.max(morphology.bounds[1])] * 3),
            )
            set_scene_range(fig.layout.scene, rng, swapaxes=swapaxes)
            set_scene_aspect(fig.layout.scene, rng, swapaxes=swapaxes)
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
            raise ArgumentError("Missing required keyword argument `input_region`.")
        return r

    return wrapper_function


def _plot_network(network, fig, cubic, swapaxes):
    xmin, xmax, ymin, ymax, zmin, zmax = tuple([0] * 6)
    for type in network.cell_types.values():
        if type.entity:
            continue
        pos = type.get_placement_set().load_positions()
        if type.plotting:
            color = type.plotting.color
            opacity = type.plotting.opacity
            name = type.plotting.display_name or type.name
        else:
            color = Color(pick_for=type).hex
            opacity = 1
            name = type.name
        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1 if not swapaxes else 2],
                z=pos[:, 2 if not swapaxes else 1],
                mode="markers",
                marker=dict(color=color, size=type.spatial.radius),
                opacity=opacity,
                name=name,
            )
        )
        xmin = min(xmin, np.min(pos[:, 0], initial=0))
        xmax = max(xmax, np.max(pos[:, 0], initial=0))
        ymin = min(ymin, np.min(pos[:, 1], initial=0))
        ymax = max(ymax, np.max(pos[:, 1], initial=0))
        zmin = min(zmin, np.min(pos[:, 2], initial=0))
        zmax = max(zmax, np.max(pos[:, 2], initial=0))
    if cubic:
        rng = max(xmax - xmin, ymax - ymin, zmax - zmin)
        fig.layout.scene.xaxis.range = [xmin, xmin + rng]
        if swapaxes:
            fig.layout.scene.yaxis.range = [zmin, zmin + rng]
            fig.layout.scene.zaxis.range = [ymin, ymin + rng]
        else:
            fig.layout.scene.yaxis.range = [ymin, ymin + rng]
            fig.layout.scene.zaxis.range = [zmin, zmin + rng]


@_network_figure
def plot_network(network, fig=None, cubic=True, swapaxes=False, show=True, legend=True):
    """
    Plot a network.
    """
    _plot_network(network, fig, cubic, swapaxes)
    return fig


@_network_figure
def network_figure(fig=None, **kwargs):
    return fig


@_network_figure
def plot_detailed_network(
    network, fig=None, cubic=True, swapaxes=True, show=True, legend=True, ids=None
):
    ms = MorphologyScene(fig)
    mr = network.morphology_repository

    for cell_type in network.cell_types.values():
        segment_radius = 2.5
        m_names = cell_type.list_all_morphologies()
        ps = network.get_placement_set(cell_type.name)
        pos = ps.load_positions()
        morpho = ps.load_morphologies()
        for pos, morpho in zip(pos, morpho):
            if morpho is None:
                ms.add_soma(
                    pos,
                    color=cell_type.plotting.color,
                    soma_radius=cell_type.spatial.radius,
                )
            else:
                ms.add_morphology(
                    morpho,
                    pos,
                    color=cell_type.plotting.color,
                    soma_radius=cell_type.spatial.radius,
                    segment_radius=segment_radius,
                )
    ms.prepare_plot(swapaxes=swapaxes)
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
    swapaxes=False,
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
        fig.layout.scene.yaxis.range = range + offset[1 if not swapaxes else 2]
        fig.layout.scene.zaxis.range = range + offset[2 if not swapaxes else 1]
    return fig


def get_branch_trace(branch, color="black", width=1.0, swapaxes=False):
    if isinstance(color, dict):
        labels = branch.list_labels()
        if "soma" in labels:
            color = color["soma"]
        elif "basal_dendrites" in labels:
            color = "lightblue"
        elif "apical_dendrites" in labels:
            color = "blue"
        elif "aa_targets" in labels:
            color = "red"
        elif "pf_targets" in labels:
            color = "violet"
        elif "sc_targets" in labels:
            color = "yellow"
        elif "dendrites" in labels:
            color = "blue"
        elif "ascending_axon" in labels:
            color = "darkgreen"
        elif "parallel_fiber" in labels:
            color = "lime"
        elif "axonal_initial_segment" in labels:
            color = "lightseagreen"
        elif "axon" in labels:
            color = color["axon"]
        else:
            color = "grey"
    return go.Scatter3d(
        x=branch.points[:, 0],
        y=branch.points[:, 1 if not swapaxes else 2],
        z=branch.points[:, 2 if not swapaxes else 1],
        mode="lines",
        line=dict(width=width, color=color),
        showlegend=False,
    )


def get_soma_trace(
    soma_radius,
    offset=[0.0, 0.0, 0.0],
    color="black",
    opacity=1,
    steps=5,
    swapaxes=False,
    **kwargs,
):
    phi = np.linspace(0, 2 * np.pi, num=steps * 2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=steps)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi) * soma_radius + offset[0]
    y = np.cos(theta) * np.cos(phi) * soma_radius + offset[1 if not swapaxes else 2]
    z = np.sin(theta) * soma_radius + offset[2 if not swapaxes else 1]

    return go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        opacity=opacity,
        color=color,
        alphahull=0,
        **kwargs,
    )


@_network_figure
def plot_fiber_morphology(
    fiber,
    offset=[0.0, 0.0, 0.0],
    fig=None,
    cubic=True,
    swapaxes=False,
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
                    branch._compartments,
                    offset,
                    color=color,
                    width=segment_radius,
                    swapaxes=swapaxes,
                )
            )
            get_branch_traces(branch.child_branches, traces)

    traces = []
    get_branch_traces(fiber.root_branches, traces)

    for trace in traces:
        fig.add_trace(trace)
    return fig


@_morpho_figure
def plot_morphology(
    morphology,
    offset=None,
    fig=None,
    swapaxes=False,
    show=True,
    legend=True,
    set_range=True,
    color="black",
    reduce_branches=False,
    soma_radius=None,
    soma_opacity=1.0,
    width=1.0,
    use_last_soma_comp=True,
):
    traces = []
    for branch in morphology.branches:
        traces.append(
            get_branch_trace(branch, offset, color=color, width=width, swapaxes=swapaxes)
        )
    for trace in traces:
        fig.add_trace(trace)
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


def _get_branch_width(branch, radii):
    if isinstance(radii, dict):
        for btype in reversed(branch[-1].labels):
            if btype in radii:
                return radii[btype]
        raise Exception(
            "Plotting width not specified for branches of type " + str(branch[-1].labels)
        )
    return radii


def _get_branch_color(branch, colors):
    if isinstance(colors, dict):
        for btype in reversed(branch[-1].labels):
            if btype in colors:
                return colors[btype]
        raise Exception(
            "Plotting color not specified for branches of type " + str(branch[-1].labels)
        )
    return colors


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
        **color_args,
    )


def plotly_block_edges(origin, sizes):
    x = origin[0] + np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) * sizes[0]
    y = origin[1] + np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]) * sizes[1]
    z = origin[2] + np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]) * sizes[2]
    return go.Scatter3d(
        x=x, y=z, z=y, mode="lines", line=dict(width=1.0, color="black"), showlegend=False
    )


def set_scene_range(scene, bounds, swapaxes=False):
    if hasattr(scene, "layout"):
        scene = scene.layout.scene  # Scene was a figure
    sw1 = 1 if not swapaxes else 2
    sw2 = 2 if not swapaxes else 1
    scene.xaxis.range = [bounds[0][0], bounds[1][0]]
    scene.yaxis.range = [bounds[0][sw1], bounds[1][sw1]]
    scene.zaxis.range = [bounds[0][sw2], bounds[1][sw2]]


def set_scene_aspect(scene, bounds, mode="equal", swapaxes=True):
    if mode == "equal":
        ratios = bounds[1] - bounds[0]
        ratios = ratios / np.max(ratios)
        items = zip(["x", "y", "z"] if not swapaxes else ["x", "z", "y"], ratios)
        scene.aspectratio = dict(items)
    else:
        scene.aspectmode = mode


def set_morphology_scene_range(scene, offset_morphologies, swapaxes=False):
    """
    Set the range on a scene containing multiple morphologies.

    :param scene: A scene of the figure. If the figure itself is given, ``figure.layout.scene`` will be used.
    :param offset_morphologies: A list of tuples where the first element is offset and the 2nd is the :class:`Morphology`
    """
    min_b = np.full((len(offset_morphologies), 3), 0, dtype=float)
    max_b = np.full((len(offset_morphologies), 3), 0, dtype=float)
    for i, morpho in enumerate(offset_morphologies):
        min_b[i] = morpho[1].bounds[0]
        max_b[i] = morpho[1].bounds[1]

    x_min = np.min(min_b[:, 0])
    x_max = np.max(max_b[:, 0])
    y_min = np.min(min_b[:, 1])
    y_max = np.max(max_b[:, 1])
    z_min = np.min(min_b[:, 2])
    z_max = np.max(max_b[:, 2])

    combined_bounds = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    set_scene_range(scene, combined_bounds, swapaxes=swapaxes)


def get_morphology_range(morphology, offset=None, soma_radius=None):
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    r = soma_radius or 0.0
    itr = enumerate(morphology.flatten())
    r = [[min(min(v), -r) + offset, max(max(v), r) + offset] for i, v in itr]
    return r


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


@_figure
@_input_highlight
def plot_traces(
    traces, fig=None, show=True, legend=True, cutoff=0, range=None, x=None, **kwargs
):
    traces.order()
    subplots_fig = make_subplots(
        cols=1,
        rows=len(traces),
        subplot_titles=[trace.title for trace in traces],
        x_title="Time [ms]",
        y_title="Membrane potential [mV]",
        **kwargs,
    )
    # Save the data already in the given figure
    _data = fig.data
    for k in dir(subplots_fig):
        v = getattr(subplots_fig, k)
        if isinstance(v, types.MethodType):
            # Unbind subplots_fig methods and bind to fig.
            v = v.__func__.__get__(fig)
        fig.__dict__[k] = v
    # Restore the data
    fig.data = _data
    fig.update_layout(height=max(len(traces) * 130, 300))
    legend_groups = set()
    legends = traces.legends
    if range is not None and x is not None:
        x = np.array(x)
        x = x[cutoff:]
        mask = (x >= range[0]) & (x <= range[1])
        x = x[mask]
    for i, cell_traces in enumerate(traces):
        for j, trace in enumerate(cell_traces):
            showlegend = legends[j] not in legend_groups
            data = trace.data[cutoff:]
            if range is not None and x is not None:
                data = data[mask]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=data,
                    legendgroup=legends[j],
                    name=legends[j],
                    showlegend=showlegend,
                    mode="lines",
                    marker=dict(color=trace.color or traces.colors[j]),
                ),
                col=1,
                row=i + 1,
            )
            legend_groups.add(legends[j])

    return fig


class MorphologyScene:
    def __init__(self, fig=None):
        self.fig = fig or go.Figure()
        self._morphologies = []
        self._somas = []

    def add_morphology(self, morphology, offset=[0.0, 0.0, 0.0], **kwargs):
        self._morphologies.append((offset, morphology, kwargs))

    def add_soma(self, offset=[0.0, 0.0, 0.0], **kwargs):
        self._somas.append((offset, kwargs))

    def show(self, swapaxes=False):
        self.prepare_plot(swapaxes=swapaxes)
        self.fig.show()

    def prepare_plot(self, swapaxes=False):
        if len(self._morphologies) == 0:
            raise MorphologyError("Cannot show empty MorphologyScene")
        for o, m, k in self._morphologies:
            plot_morphology(m, offset=o, show=False, set_range=False, fig=self.fig, **k)
        for o, k in self._somas:
            self.fig.add_scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], **k)
        set_morphology_scene_range(
            self.fig.layout.scene, self._morphologies, swapaxes=swapaxes
        )
