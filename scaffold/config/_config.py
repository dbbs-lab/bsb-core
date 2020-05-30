from . import attr, list, dict, node, root
from ..objects import CellType


@root
class Configuration:
    cell_types = dict(type=CellType, required=True)

    transfer = [
        cell_types,
    ]


config_tree = {
    "cell_types": {
        "granule_cell": {
            "placement": {"class": "scaffold.placement.LayeredRandomWalk"},
            "spatial": {"radius": 3},
        },
    },
}

config = Configuration.__cast__(config_tree, None)
