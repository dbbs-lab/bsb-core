import typing

import numpy as np

from .. import config
from .._util import merge_dicts
from ..cell_types import CellType
from ..connectivity import ConnectionStrategy
from ..placement import PlacementStrategy
from ..postprocessing import AfterConnectivityHook, AfterPlacementHook
from ..simulation.simulation import Simulation
from ..storage._files import (
    CodeDependencyNode,
    MorphologyDependencyNode,
    MorphologyPipelineNode,
)
from ..storage.interfaces import StorageNode
from ..topology import Partition, Region, RegionGroup, create_topology, get_partitions
from . import types
from ._attrs import _boot_nodes, cfgdict, cfglist

if typing.TYPE_CHECKING:
    import packaging.requirements

    from ..core import Scaffold


@config.node
class NetworkNode:
    scaffold: "Scaffold"

    x: float = config.attr(type=float, required=True)
    y: float = config.attr(type=float, required=True)
    z: float = config.attr(type=float, required=True)
    origin: list[float] = config.attr(
        type=types.list(type=float, size=3),
        default=lambda: [0, 0, 0],
        call_default=True,
    )
    chunk_size: list[float] = config.attr(
        type=types.or_(
            types.list(float),
            types.scalar_expand(float, expand=lambda s: np.ones(3) * s),
        ),
        default=lambda: [100.0, 100.0, 100.0],
        call_default=True,
    )

    def boot(self):
        self.chunk_size = np.array(self.chunk_size)


@config.root
class Configuration:
    """
    The main Configuration object containing the full definition of a scaffold model.
    """

    scaffold: "Scaffold"

    name: str = config.attr()
    """
    Descriptive name of the model
    """
    components: cfglist[CodeDependencyNode] = config.list(
        type=CodeDependencyNode,
    )
    """
    List of modules relative to the project root containing extension components.
    """
    packages: cfglist["packaging.requirements.Requirement"] = config.list(
        type=types.PackageRequirement(),
    )
    """
    List of package requirement specifiers the model depends on.
    
    """
    morphologies: cfglist[MorphologyDependencyNode] = config.list(
        type=types.or_(MorphologyDependencyNode, MorphologyPipelineNode),
    )
    """
    Morphology files and processing pipelines.
    """
    storage: StorageNode = config.attr(
        type=StorageNode,
        required=True,
    )
    """
    Network storage configuration
    """
    network: NetworkNode = config.attr(
        type=NetworkNode,
        required=True,
    )
    """
    Network description
    """
    regions: cfgdict[str, Region] = config.dict(
        type=Region,
    )
    """
    Network regions
    """
    partitions: cfgdict[str, Partition] = config.dict(
        type=Partition,
        required=True,
    )
    """
    Network partitions
    """
    cell_types: cfgdict[str, CellType] = config.dict(
        type=CellType,
        required=True,
    )
    """
    Network cell types
    """
    placement: cfgdict[str, PlacementStrategy] = config.dict(
        type=PlacementStrategy,
        required=True,
    )
    """
    Network placement strategies
    """
    after_placement: cfgdict[str, AfterPlacementHook] = config.dict(
        type=AfterPlacementHook,
    )
    connectivity: cfgdict[str, ConnectionStrategy] = config.dict(
        type=ConnectionStrategy,
        required=True,
    )
    """
    Network connectivity strategies
    """
    after_connectivity: cfgdict[str, AfterConnectivityHook] = config.dict(
        type=AfterConnectivityHook,
    )
    simulations: cfgdict[str, Simulation] = config.dict(
        type=Simulation,
    )
    """
    Network simulation configuration
    """
    __module__ = "bsb.config"

    @classmethod
    def default(cls, **kwargs):
        default_args = dict(
            storage={"engine": "fs"},
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
        # Activate the scaffold property of each config node
        _boot_nodes(self, scaffold)
        self._config_isbooted = True
        # Initialise the topology from the defined regions
        regions = list(self.regions.values())
        # Arrange the topology based on network boundaries
        start = self.network.origin.copy()
        net = self.network
        end = [start[0] + net.x, start[1] + net.y, start[2] + net.z]
        # If there are any partitions not part of the topology, add them to a group
        if unmanaged := set(self.partitions.values()) - get_partitions(regions):
            p = "', '".join(p.name for p in unmanaged)
            r = scaffold.regions.add(
                "__unmanaged__", RegionGroup(children=list(unmanaged))
            )
            regions.append(r)
        scaffold.topology = create_topology(regions, start, end)

    def _update_storage_node(self, storage):
        if self.storage.engine != storage.format:
            self.storage.engine = storage.format
        if self.storage.root != storage.root:
            self.storage._root = storage.root

    def __str__(self):
        return str(self.__tree__())

    def __repr__(self):
        return f"{type(self).__qualname__}({self})"
