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
        swapaxes=True,
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
        if set_range:
            # Set the range to be the cube around the min and max of the
            # morphology
            rng = (
                np.array([np.min(morphology.bounds[0])] * 3),
                np.array([np.max(morphology.bounds[1])] * 3),
            )
            set_scene_range(fig.layout.scene, rng)
            set_scene_aspect(fig.layout.scene, rng)
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
            fig.layout.scene.yaxis.range = [ymin, ymin + rng]
            fig.layout.scene.zaxis.range = [zmin, zmin + rng]
        else:
            fig.layout.scene.yaxis.range = [ymin, ymin + rng]
            fig.layout.scene.zaxis.range = [zmin, zmin + rng]


@_network_figure
def plot_network(network, fig=None, cubic=True, swapaxes=True, show=True, legend=True):
    """
    Plot a network, either from the current cache or the storage.
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
        morpho = mr.load(m_names[0])
        for cell in cells:
            if ids is not None and cell.id not in ids:
                continue
            ms.add_morphology(
                morpho,
                cell.position,
                color=cell_type.plotting.color,
                soma_radius=cell_type.spatial.radius,
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


def get_branch_trace(branch, offset=[0.0, 0.0, 0.0], color="black", width=1.0):
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
        y=branch.points[:, 2],
        z=branch.points[:, 1],
        mode="lines",
        line=dict(width=width, color=color),
        showlegend=False,
    )


def get_soma_trace(
    soma_radius, offset=[0.0, 0.0, 0.0], color="black", opacity=1, steps=5, **kwargs
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
        **kwargs,
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


@_morpho_figure
def plot_morphology(
    morphology,
    offset=None,
    fig=None,
    swapaxes=True,
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
        traces.append(get_branch_trace(branch, offset, color=color, width=width))
    for trace in traces:
        fig.add_trace(trace)
    # return fig


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


def set_scene_range(scene, bounds):
    if hasattr(scene, "layout"):
        scene = scene.layout.scene  # Scene was a figure
    scene.xaxis.range = [bounds[0][0], bounds[1][0]]
    scene.yaxis.range = [bounds[0][2], bounds[1][2]]
    scene.zaxis.range = [bounds[0][1], bounds[1][1]]


def set_scene_aspect(scene, bounds, mode="equal", swapaxes=True):
    if mode == "equal":
        ratios = bounds[1] - bounds[0]
        ratios = ratios / np.max(ratios)
        items = zip(["x", "z", "y"] if swapaxes else ["x", "y", "z"], ratios)
        scene.aspectratio = dict(items)
    else:
        scene.aspectmode = mode


def set_morphology_scene_range(scene, offset_morphologies):
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
    set_scene_range(scene, combined_bounds)


def get_morphology_range(morphology, offset=None, soma_radius=None):
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    r = soma_radius or 0.0
    itr = enumerate(morphology.flatten())
    r = [[min(min(v), -r) + offset, max(max(v), r) + offset] for i, v in itr]
    return r


def hdf5_plot_spike_raster(
    spike_recorders,
    input_region=None,
    show=True,
    cutoff=0,
    cell_type_sort=None,
    cell_sort=None,
):
    """
    Create a spike raster plot from an HDF5 group of spike recorders.

    :param input_region: Specifies an interval ``[min, max]`` on the x axis to highlight
      as active input to the simulation.
    :type input_region: 2-element list-like
    :param show: Immediately plot the result
    :type show: bool
    :param cutoff: Amount of ms initial simulation to ignore.
    :type cutoff: float
    :param cell_type_sort: A function to sort the cell types. Must take 2 dictionaries
      as arguments, being the raster plot's x values per label and y values per label.
      Must return an array labels matching those of the x and y values to order them.
    :type cell_type_sort: function-like
    :param cell_sort: A function that takes the cell type label and set of ids and returns
     a map to sort them.
    :type cell_sort: function-like
    """
    x_labelled = {}
    y_labelled = {}
    colors = {}
    for cell_id, dataset in spike_recorders.items():
        attrs = dict(dataset.attrs)
        if len(dataset.shape) == 1 or dataset.shape[1] == 1:
            times = dataset[()] - cutoff
            set_ids = np.ones(len(times)) * int(
                attrs.get("cell_id", attrs.get("cell", cell_id))
            )
        else:
            times = dataset[:, 1] - cutoff
            set_ids = dataset[:, 0]
        label = attrs.get("label", "unlabelled")
        if not label in x_labelled:
            x_labelled[label] = []
        if not label in y_labelled:
            y_labelled[label] = []
        if not label in colors:
            colors[label] = attrs.get("color", "black")
        # Add the spike timings on the X axis.
        x_labelled[label].extend(times)
        # Set the cell id for the Y axis of each added spike timing.
        y_labelled[label].extend(set_ids)
    # Use the parallel arrays x & y to plot a spike raster
    fig = go.Figure(
        layout=dict(
            xaxis=dict(title_text="Time [ms]"), yaxis=dict(title_text="Cell [ID]")
        )
    )
    if cell_type_sort is None:
        # Sorts the cell type dictionary by cell type size
        cell_type_sort = lambda x, y: [
            k for k, v in sorted(y.items(), key=lambda kv: len(kv[1]))
        ]
    # This lambda maps each unique y value to a sorted index starting from 0
    # We define this here so that it can be used as fallback mechanism later
    _cell_sort = lambda l, sy: dict(zip(sy, np.argsort(sy)))
    if cell_sort is None:
        # If no cell sorter is given we use the fallback sorter as default sorter.
        cell_sort = _cell_sort

    sorted_labels = cell_type_sort(x_labelled, y_labelled)
    start_id = 0
    for label in sorted_labels:
        x = np.array(x_labelled[label])
        y = np.array(y_labelled[label])
        if len(y) > 0:
            uy = np.unique(y)
            # Ask the cell sorter to give a map for the unique y values. If it returns
            # something Falsy (such as None) we use the default cell sorter.
            id_map = cell_sort(label, uy) or _cell_sort(label, uy)
            len_diff = len(uy) - len(id_map)
            if len_diff > 0:
                warn(
                    f"Sorted '{label}' array do not contain all cell ids, {len_diff} {label} omitted from raster."
                )
                y_mask = np.isin(y, id_map.keys())
                y = y[y_mask]
                x = x[x_mask]
            # Build a new numpy array using the `id_map` dictionary lookup
            y = np.vectorize(id_map.__getitem__)(y) + start_id
            start_id += len(uy)
        plot_spike_raster(
            x,
            y,
            label=label,
            fig=fig,
            show=False,
            color=colors[label],
            input_region=input_region,
        )
    fig.update_layout(xaxis=dict(range=[0, np.max(x, initial=0)]))
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
            **kwargs,
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
        # If an element of `groups` point to a single set, rather than a group
        # catch the exception and construct a single element group from the single set
        try:
            iter = handle[path].items()
        except AttributeError:
            target = handle[path]
            iter = ((group, target),)
            path = root
        for name, dataset in iter:
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
        self._runs = set()
        self.list = []

    def extend(self, arr, run=0):
        self.list.extend(arr[:, 1])
        self._runs.add(run)

    @property
    def runs(self):
        return len(self._runs)


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
def hdf5_plot_psth(
    network, handle, duration=3, cutoff=0, start=0, fig=None, gaps=True, **kwargs
):
    psth = PSTH()
    row_map = {}
    for g in handle.values():
        l = g.attrs.get("label", "unlabelled")
        cts = g.attrs.get("cell_types", [])
        color = None
        if cts:
            if len(cts) > 1:
                warn(
                    "Multiple cell types detected in a single dataset, can't perform proper PSTH"
                )
            ct = network.configuration.cell_types[cts[0]]
            l = ct.plotting.label
            color = ct.plotting.color
        elif l in network.configuration.cell_types:
            ct = network.configuration.cell_types[l]
            l = ct.plotting.label
            color = ct.plotting.color
        if l not in row_map:
            color = g.attrs.get("color", color)
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
        x_title=kwargs.get("x_title", "Time [ms]"),
        y_title=kwargs.get("y_title", "Population firing rate [Hz]"),
    )
    for k in dir(subplots_fig):
        if k == "data" or k == "_data":
            # Don't overwrite data already on the fig
            continue
        v = getattr(subplots_fig, k)
        if isinstance(v, types.MethodType):
            # Unbind subplots_fig methods and bind to fig.
            v = v.__func__.__get__(fig)
        fig.__dict__[k] = v
    # Align xaxis ranges to max of all rows
    _max = -float("inf")
    for i, row in enumerate(psth.rows):
        _max = max(_max, row.max)
    fig.update_xaxes(range=[start, _max])
    fig.update_layout(title_text=kwargs.get("title", "PSTH"))
    if not gaps:
        fig.update_layout(bargap=0, bargroupgap=0)
    cell_types = network.get_cell_types()
    for i, row in enumerate(psth.ordered_rows()):
        for name, stack in sorted(row.stacks.items(), key=lambda x: x[0]):
            counts, bins = np.histogram(stack.list, bins=np.arange(start, _max, duration))
            # Workaround of Workarounds for merging info in scaffold and in results
            # Compares plotting colors to identify cell type ...
            for cell_type in cell_types:
                if cell_type.plotting.color.lower() == stack.color:
                    current_cell_type = cell_type
                    break
            else:
                raise Exception(
                    f"Couldn't link result group '{name or row.name}' to a network cell type."
                )
            cell_num_single_run = network.get_placed_count(current_cell_type.name)
            cell_num = cell_num_single_run * (stack.runs)
            if str(name).startswith("##"):
                # Lazy way to order the stacks; Stack names can start with ## and a number
                # and it will be sorted by name, but the ## and number are not displayed.
                name = name[4:]
            bar_kwargs = dict()
            if not gaps:
                bar_kwargs["marker_line_width"] = 0
            trace = go.Bar(
                x=bins,
                y=counts / cell_num * 1000 / duration,
                name=name or row.name,
                marker=dict(color=stack.color),
                **bar_kwargs,
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
            plot_morphology(m, offset=o, show=False, set_range=False, fig=self.fig, **k)
        set_morphology_scene_range(self.fig.layout.scene, self._morphologies)
