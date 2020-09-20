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
            hex_color = cell_type.plotting.color.lstrip("#")
            color = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
            if hasattr(cell_type.plotting, "opacity"):
                color += (cell_type.plotting.opacity,)
            else:
                color += (1.0,)
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
