import math

try:
    import bpy
except ImportError:
    raise ImportError(
        "The blender module can only be used from inside the Blender environment."
    )


def create_collection(name, parent=None):
    """
    Create a collection in the blender scene.
    """
    coll_diff = _diffkey(bpy.data.collections)
    bpy.ops.collection.create(name=name)
    coll = coll_diff().pop()
    if parent is not None:
        parent.children.link(coll)
    return coll


class BlenderPopulation:
    def __init__(self, collection, cells):
        self.name = ".".join(collection.name.split(".")[:-1])
        self.collection = collection
        self.cells = cells


def create_population(name, material, cells, parent=None, scene=None, radius=3.0):
    """
    Create a cell population where each cell is represented by a sphere in 3D space.
    Each cell population will have a matte material associated with it.
    """
    if scene is None:
        scene = bpy.context.scene
    collection = create_collection(name, parent=parent)
    mesh = _create_ico_mesh(name, radius)
    mesh.materials.append(material)
    for i, cell in enumerate(cells):
        cell.object = object = bpy.data.objects.new(
            name=f"{name} #{cell.id}", object_data=mesh
        )
        object["cell_id"] = cell.id
        object.location = cell.position[[0, 2, 1]]
        collection.objects.link(object)
    return BlenderPopulation(collection, cells)


def create_material(name, color=(0.8, 0.8, 0.8, 1.0)):
    """
    Create a material with a certain base color. The 4th float of the color is the
    opacity.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principal = mat.node_tree.nodes[0]
    principal.inputs["Base Color"].default_value = color
    principal.inputs["Alpha"].default_value = color[3]
    mat.diffuse_color = color
    mat.blend_method = "BLEND"

    return mat


def create_pulsar_material(name, color, max_intensity=1.0):
    """
    Create a material capable of lighting up.
    """
    mat = bpy.data.materials.new(name=name)
    mat["pulsar"] = True
    mat.use_nodes = True
    mat.blend_method = "BLEND"

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.remove(nodes[0])

    output_node = nodes[0]
    emit_node = nodes.new("ShaderNodeEmission")
    object_node = nodes.new("ShaderNodeObjectInfo")
    math_nodes = [
        nodes.new("ShaderNodeMath"),
        nodes.new("ShaderNodeMath"),
        nodes.new("ShaderNodeMath"),
    ]
    mix_node = nodes.new("ShaderNodeMixShader")
    transparency_node = nodes.new("ShaderNodeBsdfTransparent")

    emit_node.inputs["Color"].default_value = color
    emit_node.inputs["Strength"].default_value = 12
    for m in math_nodes:
        m.operation = "MULTIPLY"
    math_nodes[2].inputs[1].default_value = max_intensity / 2

    links.new(object_node.outputs["Color"], math_nodes[0].inputs[0])
    links.new(object_node.outputs["Color"], math_nodes[0].inputs[1])
    links.new(math_nodes[0].outputs[0], math_nodes[1].inputs[0])
    links.new(math_nodes[0].outputs[0], math_nodes[1].inputs[1])
    links.new(math_nodes[1].outputs[0], math_nodes[2].inputs[0])
    links.new(math_nodes[2].outputs[0], mix_node.inputs[0])
    links.new(transparency_node.outputs[0], mix_node.inputs[1])
    links.new(emit_node.outputs["Emission"], mix_node.inputs[2])
    links.new(mix_node.outputs[0], output_node.inputs["Surface"])

    return mat


def get_population(collection, cells, partial=False):
    """
    Load or create a collection from a certain collection. Returns the loaded objects.
    """
    index = {int(c["cell_id"]): c for c in collection.objects.values()}
    if partial:
        ncells = []
        for cell in cells:
            try:
                cell.object = index[cell.id]
                ncells.append(cell)
            except KeyError:
                pass
        cells = ncells
    else:
        for cell in cells:
            try:
                cell.object = index[cell.id]
            except KeyError:
                raise Exception(
                    f"Cell {cell.id} missing from collection {collection.name}"
                )
    return BlenderPopulation(collection, cells)


def get_populations(collections, cells, partial=False):
    """
    Zips a list of collections and a list of cell lists and passes them to
    `get_population`. Returns the results as a list.
    """
    return [get_population(c, p) for c, p in zip(collections, cells, partial=partial)]


def _diffkey(coll):
    old_keys = set(coll.keys())

    def diff():
        nonlocal old_keys
        new_keys = set(coll.keys())
        diff = new_keys - old_keys
        old_keys = new_keys
        return {coll[d] for d in diff}

    return diff


def _create_ico_mesh(name, radius):
    mesh_diff = _diffkey(bpy.data.meshes)
    obj_diff = _diffkey(bpy.data.objects)
    m = max(math.ceil(math.sqrt(radius)) - 1, 1)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=m, radius=radius)
    bpy.ops.object.shade_smooth()
    mesh = mesh_diff().pop()
    mesh.name = name
    obj = obj_diff().pop()
    bpy.context.scene.collection.objects.unlink(obj)
    return mesh


def _report(message="", title="BSB Framework", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(
        draw, title=f"{title} - {icon.title()}", icon=icon
    )


def compose():
    bpy.context.scene.render.engine = "CYCLES"
