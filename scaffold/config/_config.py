from . import attr, list, dict, node, root, pluggable, on, after, before
from ..objects import CellType, Layer, Region
from . import types
from ._make import walk_nodes
from ._hooks import run_hook, has_hook
from .nodes import StorageNode, NetworkNode
from ..storage import get_engines
from ..connectivity import ConnectionStrategy
from ..simulation import SimulatorAdapter
from ..postprocessing import PostProcessingHook
import os


@root
class Configuration:
    name = attr()
    storage = attr(type=StorageNode, required=True)
    network = attr(type=NetworkNode, required=True)
    regions = dict(type=Region)
    layers = dict(type=Layer, required=True)
    cell_types = dict(type=CellType, required=True)
    after_placement = dict(type=PostProcessingHook)
    connection_types = dict(type=ConnectionStrategy, required=True)
    after_connectivity = dict(type=PostProcessingHook)
    simulations = dict(type=SimulatorAdapter)

    @classmethod
    def default(cls):
        conf = cls.__cast__(
            {
                "storage": {"engine": "hdf5"},
                "network": {"x": 200, "z": 200},
                "cell_types": {},
                "layers": {},
                "connection_types": {},
            },
            None,
        )
        conf._meta = None
        conf._parser = "json"
        conf._file = None
        return conf

    def _bootstrap(self, scaffold):
        for node in walk_nodes(self):
            node.scaffold = scaffold
            run_hook(node, "boot")
