from . import attr, list, dict, node, root
from ..objects import CellType, Layer


@root
class Configuration:
    cell_types = dict(type=CellType, required=True)
    layers = dict(type=Layer, required=True)


config_tree = {
    "layers": {"granular_layer": {"thickness": 150}},
    "cell_types": {
        "granule_cell": {
            "placement": {
                "class": "scaffold.placement.LayeredRandomWalk",
                "layer": "granular_layer",
            },
            "spatial": {"radius": 3},
        },
    },
}

# Load:
# 1) Ask the tree from the ConfigParser
# 2) Cast ourselves using the tree
# 3) Resolve reference attributes
config = Configuration.__cast__(config_tree, None)
