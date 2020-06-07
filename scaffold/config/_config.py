from . import attr, list, dict, node, root
from ..objects import CellType, Layer
from . import validators, types
from ..storage import get_engines


@node
class StorageNode:
    engine = attr(required=True, validation=validators.in_(get_engines().keys()))
    root = attr(type=types.any)


@node
class NetworkNode:
    x = attr(type=float, required=True)
    z = attr(type=float, required=True)


@node
class LayerStack:
    name = attr(key=True)
    origin = list(type=float, size=3, required=True)


@root
class Configuration:
    storage = attr(type=StorageNode, required=True)
    network = attr(type=NetworkNode, required=True)
    stacks = dict(type=LayerStack)
    layers = dict(type=Layer, required=True)
    cell_types = dict(type=CellType, required=True)
