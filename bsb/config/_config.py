from typing import TYPE_CHECKING

from . import attr, list, dict, root, node, types, provide as config_property
from ..cell_types import CellType
from ._attrs import _boot_nodes, file as file_attr
from ..placement import PlacementStrategy
from ..storage.interfaces import StorageNode
from ..connectivity import ConnectionStrategy
from ..simulation.simulation import Simulation
from ..postprocessing import PostProcessingHook
from ..exceptions import UnmanagedPartitionError, CodeImportError
from .._util import merge_dicts
from ..topology import (
    get_partitions,
    create_topology,
    Region,
    Partition,
)
import builtins
import numpy as np
import os
import sys

if TYPE_CHECKING:
    from ..storage._files import FileDependency


@node
class NetworkNode:
    x = attr(type=float, required=True)
    y = attr(type=float, required=True)
    z = attr(type=float, required=True)
    origin = attr(
        type=types.list(type=float, size=3), default=lambda: [0, 0, 0], call_default=True
    )
    chunk_size = attr(
        type=types.or_(
            types.list(float),
            types.scalar_expand(float, expand=lambda s: np.ones(3) * s),
        ),
        default=lambda: [100.0, 100.0, 100.0],
        call_default=True,
    )

    def boot(self):
        self.chunk_size = np.array(self.chunk_size)


@node
class FileDependencyNode:
    file: "FileDependency" = file_attr()

    def __init__(self, value=None, **kwargs):
        if value is not None:
            self.file = value

    def __inv__(self):
        if self._config_pos_init:
            return self.file._given_source
        else:
            return self.__tree__()


@node
class CodeDependencyNode(FileDependencyNode):
    module: str = attr(type=str)

    @config_property
    def file(self):
        from ..storage._files import FileDependency

        return FileDependency(self.module.replace(".", os.sep) + ".py")

    def __init__(self, module=None, **kwargs):
        super().__init__(**kwargs)
        if module is not None:
            self.module = module

    def load_object(self):
        import importlib.util
        import sys

        sys.path.append(os.getcwd())
        try:
            with self.file.provide_locally() as path:
                spec = importlib.util.spec_from_file_location(self.module, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[self.module] = module
                spec.loader.exec_module(module)
        finally:
            tmp = builtins.list(reversed(sys.path))
            tmp.remove(os.getcwd())
            sys.path = builtins.list(reversed(tmp))


@root
class Configuration:
    """
    The main Configuration object containing the full definition of a scaffold model.
    """

    name = attr()
    components = list(type=CodeDependencyNode)
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
    __module__ = "bsb.config"

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


def _bootstrap_components(components, file_store=None):
    for component in components:
        component_node = CodeDependencyNode(component)
        component_node.file_store = file_store
        component_node.load_object()
