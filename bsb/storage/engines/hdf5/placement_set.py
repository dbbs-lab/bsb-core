from ....models import Cell
from ....exceptions import *
from .resource import Resource
from ...interfaces import PlacementSet as IPlacementSet, MorphologySet
from .chunks import ChunkLoader, ChunkedProperty
import numpy as np

_pos_prop = lambda l: ChunkedProperty(l, "position", shape=(0, 3), dtype=float)
_rot_prop = lambda l: ChunkedProperty(l, "rotation", shape=(0, 3), dtype=float)
_morpho_prop = lambda l: ChunkedProperty(l, "morphology", shape=(0, 1), dtype=int)


class PlacementSet(
    Resource,
    IPlacementSet,
    ChunkLoader,
    properties=(_pos_prop, _rot_prop, _morpho_prop),
    collections=("labels", "additional"),
):
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
        ChunkLoader.__init__(self)
        self.type = cell_type
        self.tag = tag

    @classmethod
    def create(cls, engine, cell_type):
        """
        Create the structure for this placement set in the HDF5 file. Placement sets are
        stored under ``/cells/placement/<tag>``.
        """
        root = "/cells/placement/"
        tag = cell_type.name
        path = root + tag
        with engine._write() as fence:
            with engine._handle("a") as h:
                g = h.create_group(path)
                chunks = g.create_group("chunks")
        return cls(engine, cell_type)

    @staticmethod
    def exists(engine, cell_type):
        with engine._read():
            with engine._handle("r") as h:
                return "/cells/placement/" + cell_type.name in h

    @classmethod
    def require(cls, engine, cell_type):
        root = "/cells/placement/"
        tag = cell_type.name
        path = root + tag
        with engine._write():
            with engine._handle("a") as h:
                g = h.require_group(path)
                chunks = g.require_group("chunks")
        return cls(engine, cell_type)

    def load_positions(self):
        """
        Load the cell positions.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        try:
            return self._position_chunks.load()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No position information for the '{}' placement set.".format(self.tag)
            )

    def load_rotations(self):
        """
        Load the cell rotations.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        try:
            return self._rotation_chunks.load()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No rotation information for the '{}' placement set.".format(self.tag)
            )

    def load_morphologies(self):
        """
        Load the cell morphologies.

        :raises: DatasetNotFoundError when there is no morphology information for this
           cell type.
        """
        try:
            return MorphologySet(
                self._morphology_chunks.load(), self._get_morphology_map()
            )
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No morphology information for the '{}' placement set.".format(self.tag)
            )

    def _get_morphology_map(self):
        with self._engine._read():
            with self._engine._handle("r") as f:
                return f().attrs.get("morphology_map", [])

    def _set_morphology_map(self, map):
        with self._engine._read():
            with self._engine._handle("r") as f:
                f().attrs["morphology_map"] = map

    def load_cells(self):
        """
        Reorganize the available datasets into a collection of :class:`Cells
        <.models.Cell>`
        """
        return [Cell(i, self.type, *data) for id, data in enumerate(self)]

    def __iter__(self):
        return zip(
            self._none(self.load_positions()),
            self._none(self.load_rotations()),
        )

    def __len__(self):
        return sum(self._identifier_chunks.load(raw=True))

    def _none(self, starter):
        """
        Yield from ``starter`` then start yielding ``None``
        """
        yield from starter
        while True:
            yield None

    def append_data(
        self, chunk, positions=None, rotations=None, morphologies=None, additional=None
    ):
        """
        Append data to the PlacementSet.

        :param positions: Cell positions
        :type positions: ndararray
        :param rotations: Cell rotations
        :type rotations: ndararray
        :param morphologies: An ndarray of strings (morphology names), ndarray
          of ints (mapped morphology names), or a MorphologySet.
        :type morphologies: ndarray or class:`~.storage.interfaces.MorphologySet`
        """
        if positions is not None:
            self._position_chunks.append(chunk, positions)
        if rotations is not None:
            self._rotation_chunks.append(chunk, rotations)
        if morphologies is not None:
            self._append_morphologies(chunk, morphologies)

    def append_data(self, chunk, count, additional=None):
        """
        Append entities to the PlacementSet.

        :param positions: Cell positions
        :type positions: ndararray
        :param rotations: Cell rotations
        :type rotations: ndararray
        :param morphologies: An ndarray of strings (morphology names), ndarray
          of ints (mapped morphology names), or a MorphologySet.
        :type morphologies: ndarray or class:`~.storage.interfaces.MorphologySet`
        """
        with self._engine._write():
            with self._engine._handle("a") as f:
                n = f()

    def _append_morphologies(self, chunk, morphologies):
        dtype = getattr(morphologies, "dtype", None)
        morphology_set = self.load_morphologies()
        if "str" in dtype:
            new_set = MorphologySet.map_data(morphologies)
            morphology_set = morphology_set.merge(new_set)
        elif "int" in dtype:
            morphology_set.insert(morphologies)
        elif isinstance(morphologies, MorphologySet):
            morphology_set = morphology_set.merge(new_set)
        else:
            raise PlacementError(f"Invalid morphology data.")
        self._set_morphology_map(morphology_set.map)
        self._morphology_chunks.clear(chunk)
        self._morphology_chunks.append(chunk, morphology_set.data)

    def append_cells(self, cells):
        for cell in cells:
            raise NotImplementedError("Sorry. Not added yet.")

    def create_additional(self, name, chunk, data):
        with self._engine._write():
            with self._engine._handle("a") as f:
                path = self._path + "/additional/" + name
                maxshape = list(data.shape)
                maxshape[0] = None
                f().create_dataset(path, data=data, maxshape=tuple(maxshape))
