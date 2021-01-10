"""
The chunks module provides the tools for the HDF5 engine to store the chunked placement
data received from the placement module in seperate datasets to arbitrarily parallelize
and scale scaffold models.

The module provides the :class:`.ChunkLoader` mixin for
:class:`~.storage.engines.hdf5.resource.Resource` objects (e.g. PlacementSet,
ConnectivitySet) to organize :class:`.ChunkedProperty` and :class:`.ChunkedCollection`
objects within them.
"""

from .resource import Resource
import numpy as np


class ChunkLoader:
    """
    :class:`~.storage.engines.hdf5.resource.Resource` mixin to organize chunked properties
    and collections within itself.

    :param properties: An iterable of functions that construct :class:`.ChunkedProperty`.
    :type: iterable
    :param properties: An iterable of names for constructing :class:`.ChunkedCollection`.
    :type: iterable
    """

    def __init_subclass__(cls, properties=(), collections=(), **kwargs):
        super().__init_subclass__(**kwargs)
        cls._properties = list(properties)
        cls._collections = list(collections)

    def __init__(self):
        self._chunks = set()
        self._properties = []
        self._collections = []
        for prop_constr in self.__class__._properties:
            prop = prop_constr(self)
            print("setting", f"_{prop.name}_chunks")
            self.__dict__[f"_{prop.name}_chunks"] = prop
            self._properties.append(prop)
        for col_name in self.__class__._collections:
            col = ChunkedCollection(self, col_name)
            self.__dict__[f"_{col}_chunks"] = col
            self._collections.append(col)

    def get_chunk_path(self, chunk=None):
        """
        Return the full HDF5 path of a chunk.

        :param chunk: Chunk (e.g. ``(0, 0, 0)``)
        :type chunk: iterable
        :returns: HDF5 path
        :rtype: str
        """
        if chunk is None:
            return self._path + "/chunks/"
        return self._path + "/chunks/" + self.get_chunk_tag(chunk)

    def get_chunk_tag(self, chunk):
        """
        Return the base name of a chunk inside the HDF5 file.

        :param chunk: Chunk (e.g. ``(0, 0, 0)``)
        :type chunk: iterable
        :returns: Chunk name (e.g. ``"0.0.0"``)
        :rtype: str
        """
        return ".".join(map(lambda c: str(int(float(c))), chunk))

    def load_chunk(self, chunk):
        """
        Add a chunk to read data from when loading properties/collections.
        """
        self._chunks.add(chunk)

    def unload_chunk(self, chunk):
        """
        Remove a chunk to read data from when loading properties/collections.
        """
        self._chunks.discard(chunk)

    def require_chunk(self, chunk):
        """
        Create a chunk if it doesn't exist yet, or do nothing.
        """
        with self._engine.open("a") as f:
            path = self.get_chunk_path(chunk)
            if path in f():
                return
            chunk_group = f().create_group(path)
            for p in self._properties:
                chunk_group.create_dataset(
                    f"{path}/{p.name}", p.shape, maxshape=p.maxshape, dtype=p.dtype
                )
            for c in self._collections:
                chunk_group.create_group(path + f"/{c.name}")


class ChunkedProperty:
    """
    Chunked properties are stored inside the ``chunks`` group of the :class:`.ChunkLoader`
    they belong to. Inside the ``chunks`` group another group is created per chunk, inside
    of which a dataset exists per property.
    """

    def __init__(self, loader, property, shape, dtype, insert=None, extract=None):
        self.loader = loader
        self.name = property
        self.dtype = dtype
        self.shape = shape
        self.insert = insert
        self.extract = extract
        maxshape = list(shape)
        maxshape[0] = None
        self.maxshape = tuple(maxshape)

    def load(self):
        if not self.loader._chunks:
            chunk_group = Resource(self.loader._engine, self.loader._path + "/chunks")
            self.loader._chunks = set(
                tuple(int(x) for x in k.split(".")) for k in chunk_group.keys()
            )
        chunk_loader = map(self.read_chunk, self.loader._chunks)
        # Concatenate all non-empty chunks together
        chunked_data = tuple(c for c in chunk_loader if c.size)
        if not len(chunked_data):
            return np.empty(0)
        return np.concatenate(chunked_data)

    def read_chunk(self, chunk):
        self.loader.require_chunk(chunk)
        with self.loader._engine.open("r") as f:
            chunk_group = f()[self.loader.get_chunk_path(chunk)]
            if self.name not in chunk_group:
                return np.empty(self.shape)
            data = chunk_group[self.name][()]
            if self.extract is not None:
                data = self.extract(data)
                # Allow only `np.ndarray`. Sorry things that quack, today we're checking
                # birth certificates. Purebred ducks only.
                if type(data) is not np.ndarray:
                    # Just kidding, as long as you quack you're welcome, but you'll have
                    # to change your family name.
                    data = np.array(data)
            return data

    def append(self, chunk, data):
        if self.insert is not None:
            data = self.insert(data)
        self.loader.require_chunk(chunk)
        with self.loader._engine.open("a") as f:
            chunk_group = f()[self.loader.get_chunk_path(chunk)]
            if self.name not in chunk_group:
                chunk_group.create_dataset(
                    self.name,
                    self.shape,
                    data=data,
                    maxshape=self.maxshape,
                    dtype=self.dtype,
                )
            else:
                dset = chunk_group[self.name]
                start_pos = dset.shape[0]
                dset.resize(start_pos + len(data), axis=0)
                print(dset[start_pos:].shape, len(data), type(data), data)
                dset[start_pos:] = data


class ChunkedCollection:
    """
    Chunked collections are stored inside the ``chunks`` group of the
    :class:`.ChunkLoader` they belong to. Inside the ``chunks`` group another group is
    created per chunk, inside of which a group exists per collection. Arbitrarily named
    datasets can be stored inside of this collection.
    """

    def __init__(self, loader, property):
        self.loader = loader
        self.name = property
