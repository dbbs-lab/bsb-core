from ..exceptions import *
import abc
from ..exceptions import *
from ..reporting import report, warn
from .. import config
from ..config import refs, types
import numpy as np


@config.dynamic
class PlacementStrategy(abc.ABC):
    partitions = config.reflist(refs.partition_ref, required=True)
    density = config.attr(type=float)
    planar_density = config.attr(type=float)
    placement_count_ratio = config.attr(type=float)
    density_ratio = config.attr(type=float)
    placement_relative_to = config.ref(refs.cell_type_ref)
    count = config.attr(type=int)

    def __boot__(self):
        self.cell_type = self._config_parent

    @abc.abstractmethod
    def place(self, chunk):
        pass

    def queue(self, pool, chunk_size):
        for p in self.partitions:
            chunks = p.to_chunks(chunk_size)
            for chunk in chunks:
                pool.queue_placement(self.cell_type, chunk)

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    def get_placement_count(self):
        return int(sum(p.volume * (self.density or 0.000001) for p in self.partitions))


@config.node
class FixedPositions(PlacementStrategy):
    positions = config.attr(type=np.array)

    def place(self):
        self.scaffold.place_cells(self.cell_type, self.positions)

    def get_placement_count(self):
        return len(self.positions)


class Entities(PlacementStrategy):
    """
    Implementation of the placement of entities (e.g., mossy fibers) that do not have
    a 3D position, but that need to be connected with other cells of the scaffold.
    """

    entities = True

    def place(self):
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
