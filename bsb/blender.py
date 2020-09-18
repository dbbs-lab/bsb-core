import sys, types, math

try:
    import bpy, bpy_types, bmesh
except ImportError:
    raise ImportError(
        "The blender module can only be used from inside the Blender environment."
    )


def create_network(scaffold, scene, name):
    """
        Creates the root collection that will contain all the blender components of this
        network and a child collection for the cell populations. Fills the scene with a
        default camera and light if they are missing.
    """
    scaffold._blender_scene = scene
    scaffold._blender_collection = coll = create_collection(None, name)
    scaffold._blender_name = coll.name
    scene.collection.children.link(coll)
    cells = scaffold.create_collection("cells", parent=scaffold._blender_collection)
    scaffold._blender_cells_collection = cells
    scaffold.create_network_cam()
    scaffold.create_lighting()
    return coll


def _load_blender_network(scaffold, scene, root_collection):
    scaffold._blender_scene = scene
    scaffold._blender_collection = root_collection
    scaffold._blender_name = root_collection.name
    scaffold._blender_cells_collection = root_collection.children["cells"]


def blend(scaffold, scene, name):
    """
        Create or load the network from the given scene.
    """
    if name in scene.collection.children:
        _load_blender_network(scaffold, scene, scene.collection.children[name])
    else:
        scaffold.create_network(scene, name)


def create_population(scaffold, tag, opacity=1):
    """
        Create a cell population where each cell is represented by a sphere in 3D space.
        Each cell population will have a material associated with it that is capable of
        lighting up, to have its activity animated.
    """
    scene = scaffold._blender_scene
    placement_set = scaffold.get_placement_set(tag)
    name = placement_set.type.name
    hex_color = placement_set.type.plotting.color.lstrip("#")
    color = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    radius = placement_set.type.placement.radius
    cells = placement_set.cells
    objects = []
    collection = scaffold.create_collection(name, scaffold._blender_cells_collection)
    mat = scaffold.create_activity_material(name, (*color, opacity))
    mesh = _create_ico_mesh(scaffold, name, radius)
    mesh.materials.append(mat)
    total = len(cells)
    for i, cell in enumerate(cells):
        cell.object = object = bpy.data.objects.new(
            name=f"{name} #{cell.id}", object_data=mesh
        )
        object.location = cell.position[[0, 2, 1]]
        collection.objects.link(object)
    return collection, cells


def has_population(scaffold, tag):
    """
        Check whether a given population of the network already exists in the scene.
    """
    return tag in scaffold._blender_cells_collection.children


def load_population(scaffold, tag, opacity=1):
    """
        Load a cell population from the scene, if it doesn't exist it is created. Couples
        the placement set cells to their blender objects.
    """
    ps = scaffold.get_placement_set(tag)
    if not ps.type.relay:
        if tag not in scaffold._blender_cells_collection.children:
            collection, cells = scaffold.create_population(tag, opacity=opacity)
        else:
            cells = ps.cells
            collection = scaffold._blender_cells_collection.children[tag]
            index = {
                int(k.split("#")[1].split(".")[0]): v
                for k, v in collection.objects.items()
            }
            for cell in cells:
                cell.object = index[cell.id]
        return collection, cells


def load_populations(scaffold):
    """
        Load all cell populations from the scene, skipping relays.
    """
    colls, cells = {}, {}
    for tag in scaffold.configuration.cell_types:
        colls[tag], cells[tag] = scaffold.load_population(tag) or (None, None)
    return colls, cells


def create_collection(scaffold, name, parent=None):
    """
        Create a collection in the blender scene.
    """
    coll_diff = _diffkey(bpy.data.collections)
    bpy.ops.collection.create(name=name)
    coll = coll_diff().pop()
    if parent is not None:
        parent.children.link(coll)
    elif scaffold is not None:
        scaffold._blender_collection.children.link(coll)
    return coll


def create_network_cam(scaffold):
    cam_data = bpy.data.cameras.new("Network Camera")
    cam_data.lens = 18
    cam = bpy.data.objects.new("Network Camera", cam_data)
    cam.location = (9.69, -10.85, 12.388)
    cam.rotation_euler = (0.6799, 0, 0.8254)
    scaffold._blender_collection.objects.link(cam)


def create_lighting(scaffold):
    if "BSB Solar" not in bpy.data.lights:
        light_data = bpy.data.lights.new(name="BSB Solar", type="SUN")
        light_data.energy = 1.5
        light_object = bpy.data.objects.new(name="BSB Solar", object_data=light_data)
        scaffold._blender_scene.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()


def create_activity_material(scaffold, name, color, max_intensity=5.5, opacity=1):
    """
        Create a material capable of lighting up.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.diffuse_color = color

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.remove(nodes[0])

    output_node = nodes[0]
    emit_node = nodes.new("ShaderNodeEmission")
    object_node = nodes.new("ShaderNodeObjectInfo")
    math_node = nodes.new("ShaderNodeMath")
    mix_node = nodes.new("ShaderNodeMixShader")
    transparency_node = nodes.new("ShaderNodeBsdfTransparent")

    emit_node.inputs["Color"].default_value = color
    math_node.operation = "MULTIPLY"
    math_node.inputs[1].default_value = max_intensity
    mix_node.inputs["Fac"].default_value = opacity

    links.new(object_node.outputs["Color"], math_node.inputs[0])
    links.new(math_node.outputs[0], emit_node.inputs["Strength"])
    links.new(transparency_node.outputs[0], mix_node.inputs[1])
    links.new(emit_node.outputs["Emission"], mix_node.inputs[2])
    links.new(mix_node.outputs[0], output_node.inputs["Surface"])

    return mat


def _diffkey(coll):
    old_keys = set(coll.keys())

    def diff():
        nonlocal old_keys
        new_keys = set(coll.keys())
        diff = new_keys - old_keys
        old_keys = new_keys
        return {coll[d] for d in diff}

    return diff


def _create_ico_mesh(scaffold, name, radius):
    mesh_diff = _diffkey(bpy.data.meshes)
    obj_diff = _diffkey(bpy.data.objects)
    m = max(math.ceil(math.sqrt(radius)) - 1, 1)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=m, radius=radius)
    bpy.ops.object.shade_smooth()
    mesh = mesh_diff().pop()
    mesh.name = name
    obj = obj_diff().pop()
    scaffold._blender_scene.collection.objects.unlink(obj)
    return mesh


def _report(message="", title="BSB Framework", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(
        draw, title=f"{title} - {icon.title()}", icon=icon
    )
