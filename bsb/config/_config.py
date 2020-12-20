from . import attr, list, dict, node, root, pluggable, on, after, before
from ..objects import CellType, Partition, Region
from . import types
from ._make import walk_nodes
from ._hooks import run_hook, has_hook
from .nodes import StorageNode, NetworkNode
from ..storage import get_engines
from ..connectivity import ConnectionStrategy
from ..simulation import Simulation
from ..postprocessing import PostProcessingHook
import os, builtins


@root
class Configuration:
    """
    The main Configuration object containing the full definition of a scaffold model.
    """

    name = attr()
    components = attr(type=builtins.dict)
    storage = attr(type=StorageNode, required=True)
    network = attr(type=NetworkNode, required=True)
    regions = dict(type=Region)
    partitions = dict(type=Partition, required=True)
    cell_types = dict(type=CellType, required=True)
    after_placement = dict(type=PostProcessingHook)
    connection_types = dict(type=ConnectionStrategy, required=True)
    after_connectivity = dict(type=PostProcessingHook)
    simulations = dict(type=Simulation)

    @classmethod
    def default(cls):
        conf = cls(
            storage={"engine": "hdf5"},
            network={"x": 200, "y": 200, "z": 200},
            cell_types={},
            partitions={},
            connection_types={},
        )
        conf._meta = None
        conf._parser = "json"
        conf._file = None
        return conf

    def _bootstrap(self, scaffold):
        for node in walk_nodes(self):
            node.scaffold = scaffold
            run_hook(node, "boot")
