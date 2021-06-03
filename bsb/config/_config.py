from . import attr, list, dict, node, root, pluggable, on, after, before
from ..objects import CellType
from . import types
from ._make import walk_nodes
from ._hooks import run_hook, has_hook
from .nodes import StorageNode, NetworkNode
from ..storage import get_engines
from ..placement import PlacementStrategy
from ..connectivity import ConnectionStrategy
from ..simulation import Simulation
from ..postprocessing import PostProcessingHook
from ..exceptions import *
import os, builtins
from ..topology import (
    get_root_regions,
    get_partitions,
    create_topology,
    Boundary,
    Region,
    Partition,
)


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
    placement = dict(type=PlacementStrategy, required=True)
    after_placement = dict(type=PostProcessingHook)
    connectivity = dict(type=ConnectionStrategy, required=True)
    after_connectivity = dict(type=PostProcessingHook)
    simulations = dict(type=Simulation)

    @classmethod
    def default(cls):
        conf = cls(
            storage={"engine": "hdf5"},
            network={"x": 200, "y": 200, "z": 200},
            partitions={},
            cell_types={},
            placement={},
            connectivity={},
        )
        conf._meta = None
        conf._parser = "json"
        conf._file = None
        return conf

    def _bootstrap(self, scaffold):
        # Transfer config to the scaffold object
        for attr_name in self._config_attrs:
            if attr_name == "storage":
                # At this point the storage has already been instantiated from the config
                # node and added to the scaffold so we should not overwrite its `storage`
                # attribute with the config node again.
                continue
            setattr(scaffold, attr_name, getattr(self, attr_name))
        # Initialise the topology from the defined regions
        regions = builtins.list(self.regions.values())
        scaffold.topology = topology = create_topology(regions)
        # If there are any partitions not part of the topology, raise an error

        if (unmanaged := set(self.partitions.values()) - get_partitions([topology])) :
            p = "', '".join(p.name for p in unmanaged)
            raise UnmanagedPartitionError(f"Please make '{p}' part of a Region.")
        # Do an initial arrangement of the topology based on network boundaries
        topology.arrange(
            Boundary([0, 0, 0], [self.network.x, self.network.y, self.network.z])
        )
        for node in walk_nodes(self):
            node.scaffold = scaffold
            run_hook(node, "boot")
