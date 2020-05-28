from .config import attr, list, dict, node, root
from .objects import CellType


@root
class Configuration:
    cell_types = dict(type=CellType, required=True)


config_tree = {
    "cell_types": {
        "granule_cell": {"jammy": 3, "tomboy": "what"},
        "stellate_cell": {"jammy": "ey"},
    },
}

config = Configuration.__cast__(config_tree, None)
