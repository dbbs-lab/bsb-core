from ..exceptions import *
import abc, itertools
from ..exceptions import *
from ..reporting import report, warn
from .. import config
from ..config import refs, types
from ..helpers import SortableByAfter
from .indicator import PlacementIndications, PlacementIndicator
import numpy as np


@config.dynamic
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
    indicator_class = PlacementIndicator

    def __boot__(self):
        self.cell_type = self._config_parent
        self._queued_jobs = []

    @abc.abstractmethod
    def place(self, chunk, chunk_size):
        """
        Central method of each placement strategy. Given a chunk, should fill that chunk
        with cells by calling the scaffold's (available as ``self.scaffold``)
        :func:`~bsb.core.Scaffold.place_cells` method.
        """
        pass

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
                job = pool.queue_placement(self, chunk, chunk_size, deps=deps)
                self._queued_jobs.append(job)

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
        return sorted(objects, key=lambda s: s.guess_cell_count())

    def guess_cell_count(self):
        return sum(ind.guess() for ind in self.get_indicators().values())

    def has_after(self):
        return hasattr(self, "after")

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        # I think the reflist should always be there.
        pass


@config.node
class FixedPositions(PlacementStrategy):
    positions = config.attr(type=np.array)

    def place(self, chunk, chunk_size):
        self.scaffold.place_cells(self.cell_type, self.positions)

    def get_placement_count(self):
        return len(self.positions)


class Entities(PlacementStrategy):
    """
    Implementation of the placement of entities (e.g., mossy fibers) that do not have
    a 3D position, but that need to be connected with other cells of the scaffold.
    """

    entities = True

    def queue(self, pool, chunk_size):
        pool.queue_placement(self, np.array([0, 0, 0]), chunk_size)

    def place(self, chunk, chunk_size):
        # Variables
        cell_type = self.cell_type
        scaffold = self.scaffold

        # Get the number of cells that belong in the available volume.
        n_cells_to_place = self.get_placement_count()
        if n_cells_to_place == 0:
            warn(
                "Volume or density too low, no '{}' cells will be placed".format(
                    cell_type.name
                ),
                PlacementWarning,
            )

        scaffold.create_entities(cell_type, n_cells_to_place)
