from ..exceptions import *
import abc, itertools
from ..exceptions import *
from ..reporting import report, warn
from .. import config
from ..config import refs, types
from ..helpers import SortableByAfter
import numpy as np


@config.dynamic
class PlacementStrategy(abc.ABC, SortableByAfter):
    """
    Quintessential interface of the placement module. Each placement strategy defines an
    approach to placing neurons into a volume.
    """

    partitions = config.reflist(refs.partition_ref, required=True)
    density = config.attr(type=float)
    planar_density = config.attr(type=float)
    placement_count_ratio = config.attr(type=float)
    density_ratio = config.attr(type=float)
    placement_relative_to = config.ref(refs.cell_type_ref)
    count = config.attr(type=int)
    after = config.reflist(refs.placement_ref)

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
                job = pool.queue_placement(self.cell_type, chunk, chunk_size, deps=deps)
                self._queued_jobs.append(job)

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    def get_indicators(self):
        """
        Return indicators per cell type. Indicators collect all configuration information
        into objects that can produce guesses as to how many cells of a type should be
        placed in a volume.
        """
        pass

    def _get_placement_count_old(self):
        if self.count is not None:
            return int(self._count_for_chunk(chunk, chunk_size, self.count))
        if self.density is not None:
            return self._density_to_count(self.density, chunk, chunk_size)
        if self.planar_density is not None:
            return self._pdensity_to_count(self.planar_density, chunk, chunk_size)
        if self.placement_relative_to is not None:
            relation = self.placement_relative_to.placement
            if self.placement_count_ratio is not None:
                count = relation.get_placement_count() * self.placement_count_ratio
                count = self._count_for_chunk(chunk, chunk_size, count)
            elif self.density_ratio:
                if relation.density is not None:
                    count = self._density_to_count(
                        relation.density * self.density_ratio, chunk, chunk_size
                    )
                elif relation.planar_density is not None:
                    count = self._pdensity_to_count(
                        relation.planar_density * self.density_ratio, chunk, chunk_size
                    )
                else:
                    raise PlacementRelationError(
                        "%cell_type.name% requires relation %relation.name% to specify density information.",
                        self.cell_type,
                        relation,
                    )
            if chunk is not None:
                # If we're checking the count for a specific chunk we give back a float so
                # that placement strategies  can use the decimal value to roll to add
                # stragglers and get better precision at high chunk counts and low
                # densities.
                return count
            else:
                # If we're checking total count round the number to an int; can't place
                # half cells.
                return int(count)

    def _density_to_count(self, density, chunk=None, size=None):
        return sum(p.volume(chunk, size) * density for p in self.partitions)

    def _pdensity_to_count(self, planar_density, chunk=None, size=None):
        return sum(p.surface(chunk, size) * planar_density for p in self.partitions)

    def _count_for_chunk(self, chunk, chunk_size, count):
        if chunk is None:
            return count
        # When getting with absolute count for a chunk give back the count
        # proportional to the volume in this chunk vs total volume
        chunk_volume = sum(p.volume(chunk, chunk_size) for p in self.partitions)
        total_volume = sum(p.volume() for p in self.partitions)
        return count * chunk_volume / total_volume

    def add_stragglers(self, chunk, chunk_size, chunk_count):
        """
        Adds extra cells when the number of cells can't be exactly divided over the number
        of chunks. Default implementation will take the ``chunk_count`` and use the
        decimal value as a random roll to add an extra cell.

        Example
        -------

        5 chunks have to place 6 cells; so each chunk is told to place 1.2 cells so each
        chunk will place 1 cell and have a 0.2 chance to place an extra straggler.

        This function will then return either 1 or 2 to each chunk that asks to add
        stragglers, depending on the outcome of the 0.2 chance roll.
        """
        return int(np.floor(chunk_count) + (np.random.rand() <= chunk_count % 1))

    @classmethod
    def get_ordered(cls, objects):
        return sorted(objects, key=lambda s: s.get_placement_count())

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
        pool.queue_placement(self.cell_type, np.array([0, 0, 0]), chunk_size)

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
