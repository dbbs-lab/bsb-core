import abc
import itertools
import typing

import numpy as np

from .. import config
from .._util import obj_str_insert
from ..config import refs, types
from ..config._attrs import cfgdict
from ..exceptions import DistributorError, EmptySelectionError
from ..mixins import HasDependencies
from ..profiling import node_meter
from ..reporting import warn
from ..storage._chunks import Chunk
from ..voxels import VoxelSet
from .distributor import DistributorsNode
from .indicator import PlacementIndications, PlacementIndicator

if typing.TYPE_CHECKING:
    from ..cell_types import CellType
    from ..core import Scaffold
    from ..topology import Partition


@config.dynamic(attr_name="strategy", required=True)
class PlacementStrategy(abc.ABC, HasDependencies):
    """
    Quintessential interface of the placement module. Each placement strategy defines an
    approach to placing neurons into a volume.
    """

    scaffold: "Scaffold"

    name: str = config.attr(key=True)
    cell_types: list["CellType"] = config.reflist(refs.cell_type_ref, required=True)
    partitions: list["Partition"] = config.reflist(refs.partition_ref, required=True)
    overrides: cfgdict["PlacementIndications"] = config.dict(type=PlacementIndications)
    depends_on: list["PlacementStrategy"] = config.reflist(refs.placement_ref)
    distribute: DistributorsNode = config.attr(
        type=DistributorsNode, default=dict, call_default=True
    )
    indicator_class = PlacementIndicator

    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("place")(cls)

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        # This comparison should sort placement strategies by name, via __repr__ below
        return str(self) < str(other)

    @obj_str_insert
    def __repr__(self):
        config_name = self.name
        if not hasattr(self, "scaffold"):
            return f"'{config_name}'"
        part_str = ""
        if len(self.partitions):
            partition_names = [p.name for p in self.partitions]
            part_str = f" into {partition_names}"
        ct_names = [ct.name for ct in self.cell_types]
        return f"'{config_name}', placing {ct_names}{part_str}"

    @abc.abstractmethod
    def place(self, chunk, indicators):
        """
        Central method of each placement strategy. Given a chunk, should fill that chunk
        with cells by calling the scaffold's (available as ``self.scaffold``)
        :func:`~bsb.core.Scaffold.place_cells` method.

        :param chunk: Chunk to fill
        :type chunk: bsb.storage._chunks.Chunk
        :param indicators: Dictionary of each cell type to its PlacementIndicator
        :type indicators: Mapping[str, bsb.placement.indicator.PlacementIndicator]
        """
        pass

    def place_cells(self, indicator, positions, chunk, additional=None):
        if additional is None:
            additional = {}
        if self.distribute._has_mdistr() or indicator.use_morphologies():
            selector_error = None
            try:
                morphologies, rotations = self.distribute._specials(
                    self.partitions, indicator, positions
                )
            except EmptySelectionError as e:
                selector_error = ", ".join(str(s) for s in e.selectors)
            if selector_error:
                # Starting from Python 3.11, even though we raise from None, the original
                # EmptySelectionError somehow still gets pickled and contains unpicklable
                # elements. So we work around by raising here, outside of the exception
                # context.
                raise DistributorError(
                    "Morphology distribution couldn't find any"
                    + f" morphologies with the following selector(s): {selector_error}"
                )
        elif self.distribute._has_rdistr():
            rotations = self.distribute(
                "rotations", self.partitions, indicator, positions
            )
            morphologies = None
        else:
            morphologies, rotations = None, None

        distr = self.distribute._curry(self.partitions, indicator, positions)
        additional.update(
            {prop: distr(prop) for prop in self.distribute.properties.keys()}
        )
        self.scaffold.place_cells(
            indicator.cell_type,
            positions=positions,
            rotations=rotations,
            morphologies=morphologies,
            additional=additional,
            chunk=chunk,
        )

    def queue(self, pool, chunk_size):
        """
        Specifies how to queue this placement strategy into a job pool. Can be overridden,
        the default implementation asks each partition to chunk itself and creates 1
        placement job per chunk.
        """
        # Get the queued jobs of all the strategies we depend on.
        deps = set(
            itertools.chain(
                *(pool.get_submissions_of(strat) for strat in self.get_deps())
            )
        )
        chunks = np.unique(
            np.concatenate([p.to_chunks(chunk_size) for p in self.partitions]), axis=0
        )
        for chunk in chunks:
            job = pool.queue_placement(self, Chunk(chunk, chunk_size), deps=deps)

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    def get_indicators(self):
        """
        Return indicators per cell type. Indicators collect all configuration information
        into objects that can produce guesses as to how many cells of a type should be
        placed in a volume.
        """
        return {
            ct.name: self.__class__.indicator_class(self, ct) for ct in self.cell_types
        }

    def guess_cell_count(self):
        return sum(ind.guess() for ind in self.get_indicators().values())

    def get_deps(self):
        return set(self.depends_on)


@config.node
class FixedPositions(PlacementStrategy):
    positions: np.ndarray = config.attr(type=types.ndarray())

    def place(self, chunk, indicators):
        if self.positions is None:
            raise ValueError(
                f"Please set `.positions` on '{self.name}' before placement."
            )
        if not len(self.positions):
            warn(f"No positions given to {self.get_node_name()}.")
            return
        for indicator in indicators.values():
            inside_chunk = VoxelSet([chunk], chunk.dimensions).inside(self.positions)
            self.place_cells(indicator, self.positions[inside_chunk], chunk)

    def guess_cell_count(self):
        if self.positions is None:
            raise ValueError(f"Please set `.positions` on '{self.name}'.")
        return len(self.positions)

    def queue(self, pool, chunk_size):
        if self.positions is None:
            raise ValueError(f"Please set `.positions` on '{self.name}'.")
        # Get the queued jobs of all the strategies we depend on.
        deps = set(
            itertools.chain(
                *(pool.get_submissions_of(strat) for strat in self.get_deps())
            )
        )
        for chunk in VoxelSet.fill(self.positions, chunk_size):
            pool.queue_placement(self, Chunk(chunk, chunk_size), deps=deps)


@config.node
class Entities(PlacementStrategy):
    """
    Implementation of the placement of entities that do not have a 3D position,
    but that need to be connected with other cells of the network.
    """

    def queue(self, pool, chunk_size):
        # Entities ignore chunks since they don't intrinsically store any data.
        pool.queue_placement(self, Chunk([0, 0, 0], chunk_size))

    def place(self, chunk, indicators):
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            # Guess total number, not chunk number, as entities bypass chunking.
            n = sum(
                # Pass the voxelset if it exists
                np.sum(indicator.guess(voxels=getattr(p, "voxelset", None)))
                for p in self.partitions
            )
            self.scaffold.create_entities(cell_type, n)


__all__ = ["Entities", "FixedPositions", "PlacementStrategy"]
