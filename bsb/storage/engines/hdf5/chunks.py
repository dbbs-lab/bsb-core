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
from ..._chunks import Chunk
import numpy as np
import contextlib


class ChunkLoader:
    """
    :class:`~.storage.engines.hdf5.resource.Resource` mixin to organize chunked properties
    and collections within itself.

    :param properties: An iterable of functions that construct :class:`.ChunkedProperty`.
    :type: Iterable
    :param properties: An iterable of names for constructing :class:`.ChunkedCollection`.
    :type: Iterable
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
            self.__dict__[f"_{prop.name}_chunks"] = prop
            self._properties.append(prop)
        for col_name in self.__class__._collections:
            col = ChunkedCollection(self, col_name)
            self.__dict__[f"_{col}_chunks"] = col
            self._collections.append(col)

    def get_loaded_chunks(self):
        if not self._chunks:
            return self.get_all_chunks()
        else:
            return self._chunks.copy()

    def get_all_chunks(self):
        with self._engine._read():
            with self._engine._handle("r") as h:
                chunks = list(h[self._path + "/chunks"].keys())
                if chunks:
                    # If any chunks have been written, this HDF5 file is tagged with a
                    # chunk size
                    size = self._get_chunk_size(h)
        return [Chunk.from_id(int(c), size) for c in chunks]

    @contextlib.contextmanager
    def chunk_context(self, *chunks):
        old_chunks = self._chunks
        self._chunks = set(chunks)
        yield
        self._chunks = old_chunks

    def get_chunk_path(self, chunk=None):
        """
        Return the full HDF5 path of a chunk.

        :param chunk: Chunk
        :type chunk: :class:`.storage.Chunk`
        :returns: HDF5 path
        :rtype: str
        """
        if chunk is None:
            return f"{self._path}/chunks/"
        else:
            return f"{self._path}/chunks/{chunk.id}"

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

    def set_chunks(self, chunks):
        self._chunks = set(chunks)

    def clear_chunks(self):
        self._chunks = set()

    def require_chunk(self, chunk, handle=None):
        """
        Create a chunk if it doesn't exist yet, or do nothing.
        """
        if handle is not None:
            self._require(chunk, handle)
        else:
            with self._engine._write():
                with self._engine._handle("a") as handle:
                    self._require(chunk, handle)

    def _require(self, chunk, handle):
        path = self.get_chunk_path(chunk)
        if path in handle:
            return
        chunk_group = handle.create_group(path)
        self._set_chunk_size(handle, chunk.dimensions)
        for p in self._properties:
            chunk_group.create_dataset(
                f"{path}/{p.name}", p.shape, maxshape=p.maxshape, dtype=p.dtype
            )
        for c in self._collections:
            chunk_group.create_group(path + f"/{c.name}")

    def clear(self, chunks=None):
        if chunks is None:
            chunks = self.get_loaded_chunks()
        for chunk in chunks:
            for prop in self._properties:
                prop.clear(chunk)

    def _set_chunk_size(self, handle, size):
        fsize = handle.attrs.get("chunk_size", size)
        if not np.allclose(fsize, size):
            raise Exception(f"Chunk size mismatch. File: {fsize}. Given: {size}")
        handle.attrs["chunk_size"] = size

    def _get_chunk_size(self, handle):
        return handle.attrs["chunk_size"]


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

    def load(self, raw=False):
        with self.loader._engine._read():
            if self.loader._chunks:
                chunks = self.loader._chunks
            else:
                chunks = self.loader.get_all_chunks()
            reader = self._chunk_reader(raw=raw)
            chunk_loader = map(reader, chunks)
            # Concatenate all non-empty chunks together
            chunked_data = tuple(c for c in chunk_loader if c.size)
        if not chunked_data:
            return np.empty(self.shape)
        return np.concatenate(chunked_data)

    def _chunk_reader(self, raw):
        """
        Create a chunk reader that either returns the raw data or extracts it.
        """

        def read_chunk(chunk):
            self.loader.require_chunk(chunk)
            with self.loader._engine._read():
                with self.loader._engine._handle("r") as f:
                    chunk_group = f[self.loader.get_chunk_path(chunk)]
                    if self.name not in chunk_group:
                        return np.empty(self.shape)
                    data = chunk_group[self.name][()]
                    return data

        # If this property has an extractor and we're not in raw mode, wrap the above
        # reader to extract the data
        if not (raw or self.extract is None):
            _f = read_chunk

            def read_chunk(chunk):
                data = self.extract(_f(chunk))
                # Allow only `np.ndarray`. Sorry things that quack, today we're checking
                # birth certificates. Purebred ducks only.
                if type(data) is not np.ndarray:
                    # Just kidding, as long as you quack you're welcome, but you'll have
                    # to change your family name.
                    data = np.array(data)
                return data

        # Return the created function
        return read_chunk

    def append(self, chunk, data):
        """
        Append data to a property chunk. Will create it if it doesn't exist.

        :param chunk: Chunk
        :type chunk: :class:`.storage.Chunk`
        """
        if self.insert is not None:
            data = self.insert(data)
        with self.loader._engine._write():
            with self.loader._engine._handle("a") as f:
                self.loader.require_chunk(chunk, handle=f)
                chunk_group = f[self.loader.get_chunk_path(chunk)]
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
                    dset[start_pos:] = data

    def clear(self, chunk):
        with self.loader._engine._write():
            with self.loader._engine._handle("a") as f:
                self.loader.require_chunk(chunk, handle=f)
                chunk_group = f[self.loader.get_chunk_path(chunk)]
                if self.name not in chunk_group:
                    chunk_group.create_dataset(
                        self.name,
                        self.shape,
                        data=np.empty(self.shape),
                        maxshape=self.maxshape,
                        dtype=self.dtype,
                    )
                else:
                    dset = chunk_group[self.name]
                    dset.resize(0, axis=0)


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
