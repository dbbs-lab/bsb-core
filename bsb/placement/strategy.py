from ..exceptions import *
import abc, itertools
from ..exceptions import *
from ..reporting import report, warn
from .. import config
from ..config import refs, types
from ..helpers import SortableByAfter
from ..morphologies import MorphologyDistributor
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
    distributor = config.attr(type=MorphologyDistributor, default=dict, call_default=True)
    indicator_class = PlacementIndicator

    def __boot__(self):
        self._queued_jobs = []

    @abc.abstractmethod
    def place(self, chunk, chunk_size, indicators):
        """
        Central method of each placement strategy. Given a chunk, should fill that chunk
        with cells by calling the scaffold's (available as ``self.scaffold``)
        :func:`~bsb.core.Scaffold.place_cells` method.
        """
        pass

    def place_cells(self, cell_type, indicator, positions, chunk):
        print("Should we place morphologies:", indicator.use_morphologies())
        if indicator.use_morphologies():
            self.place_morphologies(cell_type, indicator, positions, chunk)
        else:
            self.place_somas(cell_type, positions, chunk)

    def place_morphologies(self, cell_type, indicator, positions, chunk):
        print("Distributing morphologies")
        morphology_set = self.distributor.distribute(cell_type, indicator, positions)
        print("Distributed morphologies")
        self.scaffold.place_cells(
            cell_type, positions, morphologies=morphology_set, chunk=chunk
        )

    def place_somas(self, cell_type, positions, chunk):
        self.scaffold.place_cells(cell_type, positions, chunk=chunk)

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
            print("Queueing smth")
            chunks = p.to_chunks(chunk_size)
            for chunk in chunks:
                print("Queueing chunk")
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
        # Entities ignore chunks since they don't intrinsically store any data.
        pool.queue_placement(self, np.array([0, 0, 0]), chunk_size)

    def place(self, chunk, chunk_size, indicators):
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            # Guess total number, not chunk number, as entities bypass chunking.
            n = indicator.guess()
            if n == 0:
                warn(
                    "Volume or density too low, no '{}' cells will be placed".format(
                        cell_type.name
                    ),
                    PlacementWarning,
                )
            self.scaffold.create_entities(cell_type, n)
