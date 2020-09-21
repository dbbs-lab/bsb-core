"""
    Mixin blender module for the scaffold object. When scaffold.for_blender() is called
    all the  public callable objects of this module are bound to the instance.
"""

import bpy as _bpy
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
    cam.location = (9.69, -10.85, 12.388)
    cam.rotation_euler = (0.6799, 0, 0.8254)
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


def _pulsar(results, cells, **kwargs):
    # Frames per second
    fps = kwargs.get("fps", 60)
    # Milliseconds per second
    mps = kwargs.get("ms_per_s", 50)
    # Milliseconds per frame
    mpf = mps / fps
    # Spike duration
    spd = kwargs.get("spike_duration", 5)
    # Afterburner: fade out time after animation
    ab = kwargs.get("afterburn", 30)
    # Spike width: number of frames of rising/falling edge of spike animation
    sw = math.ceil(spd / 2 / mpf)

    def cap(l, offset=0):
        # Create the intensity sequence to animate a single spike:
        # [0, 1, 2, ... sw ..., 2, 1, 0]
        # where l is the required length of the sequence and offset the offset at the
        # start of the sequence.
        seq = [sw - abs(sw - j) for j in range(l + offset)]
        if offset:
            return seq[offset:]
        return seq

    def intensity(i, min=0.3):
        return np.ones(4) * i * (1 - min) + min

    # Swap out the objects' default material for a pulsar material
    # TODO: AAAAAAAAAAAAAAAAAAAAAAAAA

    cell_activity = {}
    for cell in cells:
        if str(cell.id) not in results:
            cell_activity[cell.id] = np.empty(0)
            continue
        cell_activity[cell.id] = activity = results[str(cell.id)][:, 1]

    last_frame = 0
    for cell in cells:
        # Hardcoded granule cell solution, fix later.
        _min = 0.3 if cell.type.name != "granule_cell" else 0.0
        spike_frames = (cell_activity[cell.id] / mpf).astype(int)
        # Last spike frame
        lsf = spike_frames[-1] if len(spike_frames) > 0 else 1
        # Create an empty intensity per frame array
        ipf_arr = np.zeros(lsf + 1)
        for frame in spike_frames:
            # Each spike overlays its intensity onto a piece of the ipf array.
            start = max(frame - sw, 0)
            outtake = ipf_arr[start : (frame + sw + 1)]
            end = min(len(ipf_arr), frame + sw + 1)
            spike_intensity = cap(sw, end - start, offset=start - frame + sw)
            # The composite is the maximum between the ipf array and the spike intensity.
            ipf_arr[start:end] = np.max((outtake, spike_intensity), axis=0)
        # Normalize the ipf array
        ipf_arr = ipf_arr / sw
        # Get the 2nd differential to find where the intensity changes direction and a
        # keyframe needs to be added to the animation of the object.
        d2 = np.nonzero(np.diff(np.diff(ipf_arr, prepend=0)))[0]

        # Frame 0 init
        cell.object.color = intensity(ipf_arr[0], _min)
        cell.object.keyframe_insert(data_path="color", frame=0)

        # Add the animation keyframes where the signal changes direction.
        for key_point in d2:
            cell.object.color = intensity(ipf_arr[key_point], _min)
            cell.object.keyframe_insert(data_path="color", frame=key_point)
        cell.object.color = np.zeros(4)

        if len(d2) > 0:
            # Store the last animation frame for later use in the collective fade out.
            last_frame = max(last_frame, d2[-1] + r)

    for cell in cells:
        _min = 0.3 if cell.type.name != "granule_cell" else 0.0
        cell.object.color = intensity(0, _min)
        cell.object.keyframe_insert(data_path="color", frame=last_frame)
        cell.object.color = np.zeros(4)
        cell.object.keyframe_insert(data_path="color", frame=last_frame + ab / 2 / mpf)

    bpy.context.scene.frame_end = last_frame + ab / mpf


def _get_type_color(type):
    hex_color = type.plotting.color.lstrip("#")
    colors = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)
    return color


def _pulsar_material_swap(cells):
    # Swaps the cell meshes' material for a pulsar one, assumes all cells of the same type
    # share the same material.
    reps = list(set([c for c in cells]))
    for cell in reps:
        type = cell.type
        mesh = cell.object.data
        has_mats = len(mesh.materials) == 0
        has_pulsar = has_mats and "pulsar" in mesh.materials[0]
        if has_mats or "pulsar" not in mesh.materials[0]:
            material = _main.create_pulsar_material(type.name, _get_type_color(type))
            if has_mats:
                mesh.materials.append(material)
            elif not has_pulsar:
                mesh.materials[0] = material


animate.pulsar = _pulsar
