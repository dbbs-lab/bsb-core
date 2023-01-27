from .. import config
from ..exceptions import (
    EmptySelectionError,
    DistributorError,
    MissingSourceError,
    SourceQualityError,
)
from ..profiling import node_meter
from ..reporting import report
from ..config import refs, types
from .._util import SortableByAfter, obj_str_insert
from ..voxels import VoxelSet
from ..storage import Chunk
from .indicator import PlacementIndications, PlacementIndicator
from .distributor import DistributorsNode
import numpy as np
import itertools
import abc


@config.dynamic(attr_name="strategy", required=True)
class PlacementStrategy(abc.ABC, SortableByAfter):
    """
    Quintessential interface of the placement module. Each placement strategy defines an
    approach to placing neurons into a volume.
    """

    name = config.attr(key=True)
    cell_types = config.reflist(refs.cell_type_ref, required=True)
    partitions = config.reflist(refs.partition_ref, required=True)
    overrides = config.dict(type=PlacementIndications)
    after = config.reflist(refs.placement_ref)
    distribute = config.attr(type=DistributorsNode, default=dict, call_default=True)
    indicator_class = PlacementIndicator

    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("place")(cls)

    def __boot__(self):
        self._queued_jobs = []

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
        """
        pass

    def place_cells(self, indicator, positions, chunk, additional=None):
        if additional is None:
            additional = {}
        if self.distribute._has_mdistr() or indicator.use_morphologies():
            try:
                morphologies, rotations = self.distribute._specials(
                    self.partitions, indicator, positions
                )
            except EmptySelectionError as e:
                selectors = ", ".join(f"{s}" for s in e.selectors)
                raise DistributorError(
                    "%property% distribution of `%strategy.name%` couldn't find any"
                    + f" morphologies with the following selector(s): {selectors}",
                    "Morphology",
                    self,
                ) from None
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
        # Reset jobs that we own
        self._queued_jobs = []
        # Get the queued jobs of all the strategies we depend on.
        deps = set(itertools.chain(*(strat._queued_jobs for strat in self.get_after())))
        for p in self.partitions:
            chunks = p.to_chunks(chunk_size)
            for chunk in chunks:
                job = pool.queue_placement(self, Chunk(chunk, chunk_size), deps=deps)
                self._queued_jobs.append(job)
        report(f"Queued {len(self._queued_jobs)} jobs for {self.name}", level=2)

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

    @classmethod
    def get_ordered(cls, objects):
        # No need to sort placement strategies, just obey dependencies.
        return objects

    def guess_cell_count(self):
        return sum(ind.guess() for ind in self.get_indicators().values())

    def has_after(self):
        return hasattr(self, "after")

    def get_after(self):
        return [] if not self.has_after() else self.after

    def create_after(self):
        # I think the reflist should always be there.
        pass


@config.node
class FixedPositions(PlacementStrategy):
    positions = config.attr(type=types.ndarray())

    def place(self, chunk, indicators):
        if self.positions is None:
            raise ValueError(
                f"Please set `.positions` on '{self.name}' before placement."
            )
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
        # Reset jobs that we own
        self._queued_jobs = []
        # Get the queued jobs of all the strategies we depend on.
        deps = set(itertools.chain(*(strat._queued_jobs for strat in self.get_after())))
        for chunk in VoxelSet.fill(self.positions, chunk_size):
            job = pool.queue_placement(self, Chunk(chunk, chunk_size), deps=deps)
            self._queued_jobs.append(job)
        report(f"Queued {len(self._queued_jobs)} jobs for {self.name}", level=2)


class Entities(PlacementStrategy):
    """
    Implementation of the placement of entities that do not have a 3D position,
    but that need to be connected with other cells of the network.
    """

    entities = True

    def queue(self, pool, chunk_size):
        # Entities ignore chunks since they don't intrinsically store any data.
        pool.queue_placement(self, Chunk([0, 0, 0], chunk_size))

    def place(self, chunk, indicators):
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            # Guess total number, not chunk number, as entities bypass chunking.
            n = indicator.guess()
            self.scaffold.create_entities(cell_type, n)
