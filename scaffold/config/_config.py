from . import attr, list, dict, node, root
from ..objects import CellType, Layer
from . import types
from ..storage import get_engines
from ..connectivity import ConnectionStrategy
from ..simulation import SimulatorAdapter
from ..postprocessing import PostProcessingHook


@node
class StorageNode:
    engine = attr(required=True, validation=types.in_(get_engines().keys()))
    root = attr(type=types.any)


@node
class NetworkNode:
    x = attr(type=float, required=True)
    z = attr(type=float, required=True)


@node
class LayerStack:
    name = attr(key=True)
    origin = attr(type=types.list(float, size=3), required=True)


@root
class Configuration:
    name = attr()
    storage = attr(type=StorageNode, required=True)
    network = attr(type=NetworkNode, required=True)
    stacks = dict(type=LayerStack)
    layers = dict(type=Layer, required=True)
    cell_types = dict(type=CellType, required=True)
    after_placement = dict(type=PostProcessingHook)
    connection_types = dict(type=ConnectionStrategy, required=True)
    after_connectivity = dict(type=PostProcessingHook)
    simulations = dict(type=SimulatorAdapter)
