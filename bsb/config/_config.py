from . import attr, list, dict, node, root, pluggable, on, after, before
from ..cell_types import CellType
from . import types
from ._attrs import _boot_nodes
from ._make import walk_nodes
from ._hooks import run_hook, has_hook
from .nodes import StorageNode, NetworkNode
from ..storage import get_engines
from ..placement import PlacementStrategy
from ..connectivity import ConnectionStrategy
from ..simulation import Simulation
from ..postprocessing import PostProcessingHook
from ..exceptions import *
from .._util import merge_dicts
import os, builtins
from ..topology import (
    get_partitions,
    create_topology,
    Region,
    Partition,
)


@root
class Configuration:
    """
    The main Configuration object containing the full definition of a scaffold model.
    """

    name = attr()
    storage = attr(type=StorageNode, required=True)
    network = attr(type=NetworkNode, required=True)
    regions = dict(type=Region)
    partitions = dict(type=Partition, required=True)
    cell_types = dict(type=CellType, required=True)
    placement = dict(type=PlacementStrategy, required=True)
    after_placement = dict(type=PostProcessingHook)
    connectivity = dict(type=ConnectionStrategy, required=True)
    after_connectivity = dict(type=PostProcessingHook)
    simulations = dict(type=Simulation)

    @classmethod
    def default(cls, **kwargs):
        default_args = builtins.dict(
            storage={"engine": "hdf5"},
            network={"x": 200, "y": 200, "z": 200},
            partitions={},
            cell_types={},
            placement={},
            connectivity={},
        )
        merge_dicts(default_args, kwargs)
        conf = cls(default_args)
        conf._parser = "json"
        conf._file = None
        return conf

    def _bootstrap(self, scaffold):
        # Initialise the topology from the defined regions
        regions = builtins.list(self.regions.values())
        # Arrange the topology based on network boundaries
        start = self.network.origin.copy()
        net = self.network
        end = [start[0] + net.x, start[1] + net.y, start[2] + net.z]
        scaffold.topology = topology = create_topology(regions, start, end)
        # If there are any partitions not part of the topology, raise an error
        if unmanaged := set(self.partitions.values()) - get_partitions([topology]):
            p = "', '".join(p.name for p in unmanaged)
            raise UnmanagedPartitionError(f"Please make '{p}' part of a Region.")
        # Activate the scaffold property of each config node
        _boot_nodes(self, scaffold)
        self._config_isbooted = True

    def _update_storage_node(self, storage):
        if self.storage.engine != storage.format:
            self.storage.engine = storage.format
        if self.storage.root != storage.root:
            self.storage.root = storage.root

    def __str__(self):
        return str(self.__tree__())

    def __repr__(self):
        return f"{type(self).__qualname__}({self})"
