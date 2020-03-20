from ....models import Cell
from ....exceptions import *
from ....helpers import (
    continuity_list,
    expand_continuity_list,
    count_continuity_list,
    iterate_continuity_list,
)
from .resource import Resource
from ...interfaces import PlacementSet as IPlacementSet
import numpy as np


class PlacementSet(Resource, IPlacementSet):
    """
        Fetches placement data from storage. You can either access the parallel-array
        datasets ``.identifiers``, ``.positions`` and ``.rotations`` individually or
        create a collection of :class:`Cells <.models.Cell>` that each contain their own
        identifier, position and rotation.

        .. note::

            Use :func:`.core.get_placement_set` to correctly obtain a PlacementSet.
    """

    def __init__(self, engine, cell_type):
        root = "/cells/placement/"
        tag = cell_type.name
        super().__init__(engine, root + tag)
        if not self.exists(engine, cell_type):
            raise DatasetNotFoundError("PlacementSet '{}' does not exist".format(tag))
        self.type = cell_type
        self.tag = tag
        self.identifier_set = Resource(engine, root + tag + "/identifiers")
        self.positions_set = Resource(engine, root + tag + "/positions")
        self.rotation_set = Resource(engine, root + tag + "/rotations")

    @classmethod
    def create(cls, engine, cell_type):
        root = "/cells/placement/"
        tag = cell_type.name
        path = root + tag
        with engine.open("a") as h:
            g = h().create_group(path)
            g.create_dataset(path + "/identifiers", (0,), dtype=int)
            if not cell_type.entity:
                g.create_dataset(path + "/positions", (0, 3), dtype=float)
            g.create_group(path + "/additional")
        return cls(engine, cell_type)

    @staticmethod
    def exists(engine, cell_type):
        with engine.open("r") as h:
            return "/cells/placement/" + cell_type.name in h()

    @property
    def identifiers(self):
        """
            Return a list of cell identifiers.
        """
        return np.array(
            expand_continuity_list(self.identifier_set.get_dataset()), dtype=int
        )

    @property
    def positions(self):
        """
            Return a dataset of cell positions.
        """
        try:
            return self.positions_set.get_dataset()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No position information for the '{}' placement set.".format(self.tag)
            )

    @property
    def rotations(self):
        """
            Return a dataset of cell rotations.

            :raises: DatasetNotFoundError when there is no rotation information for this
               cell type.
        """
        try:
            return self.rotation_set.get_dataset()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No rotation information for the '{}' placement set.".format(self.tag)
            )

    @property
    def cells(self):
        """
            Reorganize the available datasets into a collection of :class:`Cells
            <.models.Cell>`
        """
        return [
            Cell(id, self.type, position, rotation) for id, position, rotation in self
        ]

    def __iter__(self):
        id_iter = iterate_continuity_list(self.identifier_set.get_dataset())
        iterators = [iter(id_iter), self._none(), self._none()]
        if self.positions_set.exists():
            iterators[1] = iter(self.positions)
        if self.rotation_set.exists():
            iterators[2] = iter(self.rotations)
        return zip(*iterators)

    def __len__(self):
        return count_continuity_list(self.identifier_set)

    def _none(self):
        """
            Generate ``len(self)`` times ``None``
        """
        for i in range(len(self)):
            yield None

    def append_data(self, identifiers, positions=None, rotations=None, additional=None):
        data = self.identifier_set.append(continuity_list(identifiers), dtype=int)
        if positions is not None:
            data = self.positions_set.append(positions)
        if rotations is not None:
            data = self.rotation_set.append(rotations)

    def append_cells(self, cells):
        for cell in cells:
            raise NotImplementedError("Sorry. Not added yet.")
