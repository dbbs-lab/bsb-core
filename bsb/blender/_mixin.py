"""
    Mixin blender module for the scaffold object. When scaffold.for_blender() is called
    all the  public callable objects of this module are bound to the instance.
"""

import bpy as _bpy
import numpy as _np
from .. import blender as _main


def create_network(self, scene, name):
    """
    Creates the root collection that will contain all the blender components of this
    network and a child collection for the cell populations. Fills the scene with a
    default camera and light if they are missing.
    """
    self._blender_scene = scene
    self._blender_collection = coll = _main.create_collection(name)
    self._blender_name = coll.name
    scene.collection.children.link(coll)
    cells = self.create_collection("cells", parent=self._blender_collection)
    self._blender_cells_collection = cells
    self.create_network_cam()
    self.create_lighting()
    return coll


def _load_blender_network(self, scene, root_collection):
    self._blender_scene = scene
    self._blender_collection = root_collection
    self._blender_name = root_collection.name
    self._blender_cells_collection = root_collection.children["cells"]


def blend(self, scene, name):
    """
    Create or load the network from the given scene.
    """
    if name in scene.collection.children:
        _load_blender_network(self, scene, scene.collection.children[name])
    else:
        self.create_network(scene, name)


def create_collection(self, name, parent=None):
    if parent is None:
        parent = self._blender_collection
    return _main.create_collection(name, parent=parent)


def create_population(self, tag, material=None, color=None):
    placement_set = self.get_placement_set(tag)
    cell_type = placement_set.type
    if material is None:
        if color is None:
            color = _get_type_color(cell_type)
            if hasattr(cell_type.plotting, "opacity"):
                color = color[0:3] + (cell_type.plotting.opacity,)
        material = _main.create_material(tag, color=color)
    return _main.create_population(
        tag,
        material,
        placement_set.cells,
        parent=self._blender_cells_collection,
        scene=self._blender_scene,
        radius=placement_set.type.placement.radius,
    )


def ensure_population(self, tag):
    """
    Load a cell population onto the scene, does nothing if it already exists. Entities
    are also ignored.
    """
    ps = self.get_placement_set(tag)
    if not ps.type.entity and not self.has_population(tag):
        self.create_population(tag)


def ensure_populations(self):
    """
    Load all cell populations from the scene, skipping relays.
    """
    for tag in self.configuration.cell_types:
        self.load_population(tag)


def get_population(self, tag, partial=False):
    """
    Get a cell population from the scene or create them.
    """
    type = self.get_cell_type(tag)
    if type.entity:
        return None
    if not self.has_population(tag):
        return self.create_population(tag)

    cells = self.get_placement_set(tag).cells
    collection = self._blender_cells_collection.children[tag]
    return _main.get_population(collection, cells, partial=partial)


def get_populations(self, partial=False):
    populations = {}
    for type in self.configuration.cell_types.values():
        if type.entity:
            continue
        populations[type.name] = self.get_population(type.name, partial=partial)
    return populations


def has_population(self, tag):
    """
    Check whether a given population of the network already exists in the scene.
    """
    return tag in self._blender_cells_collection.children


def create_network_cam(self):
    cam_data = _bpy.data.cameras.new("Network Camera")
    cam_data.lens = 18
    cam = _bpy.data.objects.new("Network Camera", cam_data)
    cam.location = (150.65, -331.79, 269.37)
    cam.rotation_euler = (68.3, 0, 0)
    self._blender_collection.objects.link(cam)


def create_lighting(self):
    if "BSB Solar" not in _bpy.data.lights:
        light_data = _bpy.data.lights.new(name="BSB Solar", type="SUN")
        light_data.energy = 1.5
        light_object = _bpy.data.objects.new(name="BSB Solar", object_data=light_data)
        self._blender_scene.collection.objects.link(light_object)
        _bpy.context.view_layer.objects.active = light_object
        dg = _bpy.context.evaluated_depsgraph_get()
        dg.update()


def animate():
    pass


def print_debug(lvl, cat, msg, i=None, total=None):
    if cat.startswith("cell_"):
        return
    kwargs = {}
    if i is not None:
        kwargs["end"] = "\r"
    print(f"[{cat}] {msg}", **kwargs)


def devnull(*args, **kwargs):
    pass


def _pulsar(results, cells, **kwargs):
    import math

    cells = list(cells)

    # Frames per second
    fps = kwargs.get("fps", 60)
    # Milliseconds per second
    mps = kwargs.get("ms_per_s", 25)
    # Milliseconds per frame
    mpf = mps / fps
    # Spike duration
    spd = kwargs.get("spike_duration", 5)
    # Afterburner: fade out time after animation
    ab = kwargs.get("afterburn", 30)
    # Get the listener that deals with progress reports
    listener_name = kwargs.get("listener", "devnull")
    listener = globals().get(listener_name, devnull)
    # Spike width: number of frames of rising/falling edge of spike animation
    sw = math.ceil(spd / 2 / mpf)
    # Create the signal processor functions. They calculate cell intensity during anim.
    cap, intensity = _pulsar_signal_processors(sw)
    # Swap out the cells' materials with a pulsar material that flashes based on obj color
    _pulsar_material_swap(cells)
    # Set up compositor with a glare node.
    _pulsar_glare()
    # Retrieve cell activity from the given results
    cell_activity = _pulsar_cell_activity(cells, results, listener)
    # Animate the cell keyframes
    _pulsar_animate(cells, cell_activity, mpf, sw, ab, cap, intensity, listener)


animate.pulsar = _pulsar
_crowded_pulsars = ["granule_cell", "glomerulus"]


def _pulsar_animate(cells, cell_activity, mpf, sw, ab, cap, intensity, listener):
    last_frame = 0
    for i, cell in enumerate(cells):
        # Hardcoded granule cell solution, fix later.
        _min = 0.3 if cell.type.name not in _crowded_pulsars else 0.0
        spike_frames = (cell_activity[cell.id] / mpf).astype(int)
        # Last spike frame
        lsf = spike_frames[-1] if len(spike_frames) > 0 else 1
        # Create an empty intensity per frame array
        ipf_arr = _np.zeros(lsf + sw + 1)
        for frame in spike_frames:
            # Each spike overlays its intensity onto a piece of the ipf array.
            start = max(frame - sw, 0)
            outtake = ipf_arr[start : (frame + sw + 1)]
            end = min(len(ipf_arr), frame + sw + 1)
            spike_intensity = cap(end - start, offset=start - frame + sw)
            # The composite is the maximum between the ipf array and the spike intensity.
            ipf_arr[start:end] = _np.max((outtake, spike_intensity), axis=0)
        # Get the 2nd differential to find where the intensity changes direction and a
        # keyframe needs to be added to the animation of the object.
        d2 = _np.nonzero(_np.diff(_np.diff(ipf_arr, prepend=0)))[0]
        # Normalize the ipf array
        ipf_arr = ipf_arr / sw
        # First frame
        cell.object.color = intensity(ipf_arr[0], _min)
        cell.object.keyframe_insert(data_path="color", frame=0)

        # Add the animation keyframes where the signal changes direction.
        for key_point in d2:
            cell.object.color = intensity(ipf_arr[key_point], _min)
            cell.object.keyframe_insert(data_path="color", frame=key_point)

        # Last frame
        cell.object.color = intensity(0, _min)
        cell.object.keyframe_insert(data_path="color", frame=lsf + sw)

        if len(d2) > 0:
            # Store the last animation frame for later use in the collective fade out.
            last_frame = max(last_frame, d2[-1] + sw)

    _pulsar_afterburn(cells, mpf, last_frame, ab, intensity)


def _get_type_color(type):
    hex_color = type.plotting.color.lstrip("#")
    color = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
    return color


def _pulsar_material_swap(cells):
    # Swaps the cell meshes' material for a pulsar one, assumes all cells of the same type
    # share the same material.
    reps = list(set([c for c in cells]))
    for cell in reps:
        type = cell.type
        mesh = cell.object.data
        has_no_mats = len(mesh.materials) == 0
        has_no_pulsar = not has_no_mats and "pulsar" not in mesh.materials[0]
        if has_no_mats or has_no_pulsar:
            material = _main.create_pulsar_material(type.name, _get_type_color(type))
            if has_no_mats:
                mesh.materials.append(material)
            elif has_no_pulsar:
                mesh.materials[0] = material


def _pulsar_signal_processors(sw):
    def cap(len_, offset=0):
        # Create the intensity sequence to animate a single spike:
        # [0, 1, 2, ... sw ..., 2, 1, 0]
        # where l is the required length of the sequence and offset the offset at the
        # start of the sequence.
        seq = [sw - abs(sw - j) for j in range(len_ + offset)]
        if offset:
            return seq[offset:]
        return seq

    def intensity(i, min=0.3):
        return _np.ones(4) * i * (1 - min) + min

    return cap, intensity


def _pulsar_cell_activity(cells, results, listener):
    cell_activity = {}
    for i, cell in enumerate(cells):
        listener("debug", "cell", f"id: {cell.id}", i=i, total=len(cells))
        if str(cell.id) not in results:
            listener("warn", "cell_data_not_found", f"No data for {cell.id}")
            cell_activity[cell.id] = _np.empty(0)
            continue
        activity = results[str(cell.id)][()]
        if not len(activity):
            listener("info", "cell_silent", f"Cell {cell.id} does not fire.")
            cell_activity[cell.id] = _np.empty(0)
            continue
        listener("debug", "cell_activity", f"Cell {cell.id} fires {len(activity)} times.")
        cell_activity[cell.id] = results[str(cell.id)][:, 1]
    return cell_activity


def _pulsar_glare():
    _bpy.context.scene.render.use_compositing = True
    _bpy.context.scene.use_nodes = True
    tree = _bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    if any(["pulsar" in n for n in nodes]):
        # Glare node already added, don't overwrite user customization
        return
    else:
        glare_node = _pulsar_glare_node(nodes)
        layers_node = nodes["Render Layers"]
        composite_node = nodes["Composite"]
        # Unlink default link between render layers and composite
        links.remove(links[0])
        links.new(layers_node.outputs["Image"], glare_node.inputs["Image"])
        links.new(glare_node.outputs["Image"], composite_node.inputs["Image"])


def _pulsar_glare_node(nodes):
    glare_node = nodes.new("CompositorNodeGlare")
    glare_node.glare_type = "FOG_GLOW"
    glare_node.quality = "HIGH"
    glare_node.threshold = 0.3
    glare_node.size = 8

    return glare_node


def _pulsar_afterburn(cells, mpf, last_frame, ab, intensity):
    for cell in cells:
        _min = 0.3 if cell.type.name not in _crowded_pulsars else 0.0
        cell.object.color = intensity(0, _min)
        cell.object.keyframe_insert(data_path="color", frame=last_frame + ab / 2 / mpf)
        cell.object.color = _np.zeros(4)
        cell.object.keyframe_insert(data_path="color", frame=last_frame + ab / mpf)

    _bpy.context.scene.frame_end = last_frame + ab / mpf
