from ....exceptions import *
from .resource import Resource
from ...interfaces import PlacementSet as IPlacementSet
from ....morphologies import MorphologySet
from .chunks import ChunkLoader, ChunkedProperty
import numpy as np

_pos_prop = lambda l: ChunkedProperty(l, "position", shape=(0, 3), dtype=float)
_rot_prop = lambda l: ChunkedProperty(l, "rotation", shape=(0, 3), dtype=float)
_morpho_prop = lambda l: ChunkedProperty(l, "morphology", shape=(0,), dtype=int)


class _MapSelector:
    def __init__(self, ps, names):
        self._ps = ps
        self._names = set(names)

    def validate(self, loaders):
        missing = set(self._names) - {m.get_meta()["name"] for m in loaders}
        if missing:
            raise MissingMorphologyError(
                f"Morphology repository misses the following morphologies required by {self._ps.tag}: {', '.join(missing)}"
            )

    def pick(self, stored_morphology):
        name = stored_morphology.get_meta()["name"]
        return name in self._names


class PlacementSet(
    Resource,
    ChunkLoader,
    IPlacementSet,
    properties=(_pos_prop, _rot_prop, _morpho_prop),
    collections=("labels", "additional"),
):
    """
    Fetches placement data from storage.

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

        :raises: DatasetNotFoundError when the morphology data is not found.
        """
        try:
            return MorphologySet(
                self._get_morphology_loaders(),
                self._morphology_chunks.load(),
                self._rotation_chunks.load(),
            )
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No morphology information for the '{}' placement set.".format(self.tag)
            )

    def _get_morphology_loaders(self):
        with self._engine._read():
            with self._engine._handle("r") as f:
                for chunk in self.get_loaded_chunks():
                    print("Loaded chunk:", chunk)
                    path = self.get_chunk_path(chunk)
                    _map = f[path].attrs.get("morphology_loaders", [])
                    return self._engine.morphologies.select(_MapSelector(self, _map))

    def _set_morphology_loaders(self, map):
        with self._engine._write():
            with self._engine._handle("a") as f:
                for chunk in self.get_loaded_chunks():
                    path = self.get_chunk_path(chunk)
                    f[path].attrs["morphology_loaders"] = map

    def __iter__(self):
        return zip(
            self._none(self.load_positions()),
            self._none(self.load_morphologies()),
        )

    def __len__(self):
        return sum(self._position_chunks.load(raw=True))

    def _none(self, starter):
        """
        Yield from ``starter`` then start yielding ``None``
        """
        yield from starter
        while True:
            yield None

    def append_data(
        self,
        chunk,
        positions=None,
        morphologies=None,
        additional=None,
        count=None,
    ):
        """
        Append data to the PlacementSet.

        :param positions: Cell positions
        :type positions: ndararray
        :param rotations: Cell rotations
        :type rotations: ndararray
        :param morphologies: The associated MorphologySet.
        :type morphologies: class:`~.storage.interfaces.MorphologySet`
        """
        if count is not None:
            if not (positions is None and morphologies is None):
                raise ValueError(
                    "The `count` keyword is reserved for creating entities,"
                    + " without any positional, or morphological data."
                )
            with self._engine._write():
                self.require_chunk(chunk)
                path = self.get_chunk_path(chunk)
                with self._engine._handle("a") as f:
                    prev_count = f[path].attrs.get("entity_count", 0)
                    f[path].attrs["entity_count"] = prev_count + count

        if positions is not None:
            self._position_chunks.append(chunk, positions)
        if morphologies is not None:
            self._append_morphologies(chunk, morphologies)
        if additional is not None:
            for key, ds in additional.items():
                self.append_additional(key, chunk, ds)

    def _append_morphologies(self, chunk, new_set):
        with self.chunk_context(chunk):
            morphology_set = self.load_morphologies().merge(new_set)
            self._set_morphology_loaders(morphology_set._serialize_loaders())
            self._morphology_chunks.clear(chunk)
            self._morphology_chunks.append(chunk, morphology_set.get_indices())
            self._rotation_chunks.append(chunk, new_set.get_rotations())

    def append_entities(self, chunk, count, additional=None):
        self.append_data(chunk, count=count, additional=additional)

    def append_additional(self, name, chunk, data):
        with self._engine._write():
            self.require_chunk(chunk)
            path = self.get_chunk_path(chunk) + "/additional/" + name
            with self._engine._handle("a") as f:
                if path not in f:
                    maxshape = list(data.shape)
                    maxshape[0] = None
                    f.create_dataset(path, data=data, maxshape=tuple(maxshape))
                else:
                    dset = f[path]
                    start_pos = dset.shape[0]
                    dset.resize(start_pos + len(data), axis=0)
                    dset[start_pos:] = data
