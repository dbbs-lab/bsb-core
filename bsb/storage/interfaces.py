import abc
import functools
import typing
from pathlib import Path

import numpy as np

from .. import config, plugins
from .._util import immutable, obj_str_insert
from ..trees import BoxTree
from ._chunks import Chunk

if typing.TYPE_CHECKING:
    from ..cell_types import CellType


@config.pluggable(key="engine", plugin_name="storage engine")
class StorageNode:
    root: typing.Any = config.slot()

    @classmethod
    def __plugins__(cls):
        if not hasattr(cls, "_plugins"):
            cls._plugins = {
                name: plugin.StorageNode
                for name, plugin in plugins.discover("storage.engines").items()
            }
        return cls._plugins


class Interface(abc.ABC):
    _iface_engine_key = None

    def __init__(self, engine):
        self._engine = engine

    def __init_subclass__(cls, **kwargs):
        # Only change engine key if explicitly given.
        if "engine_key" in kwargs:
            cls._iface_engine_key = kwargs["engine_key"]


class NoopLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Engine(Interface):
    """
    Engines perform the transactions that come from the storage object, and read/write
    data in a specific format. They can perform collective or individual actions.

    .. warning::

      Collective actions can only be performed from all nodes, or deadlocks occur. This
      means in particular that they may not be called from component code.

    """

    def __init__(self, root, comm):
        self._root = comm.bcast(root, root=0)
        self._comm = comm
        self._readonly = False

    def __eq__(self, other):
        eq_format = self._format == getattr(other, "_format", None)
        eq_root = self._root == getattr(other, "_root", None)
        return eq_format and eq_root

    @property
    def root(self):
        """
        The unique identifier for the storage. Usually pathlike, but can be anything.
        """
        return self._root

    @property
    def comm(self):
        """
        The communicator in charge of collective operations.
        """
        return self._comm

    def set_comm(self, comm):
        """
        :guilabel:`collective` Set a new communicator in charge of collective operations.
        """
        self._comm = comm

    @property
    def format(self):
        """
        Name of the type of engine. Automatically set through the plugin system.
        """
        return self._format

    @property
    @abc.abstractmethod
    def versions(self):
        """
        Must return a dictionary containing the version of the engine package, and bsb
        package, used to last write to this storage object.
        """
        pass

    @property
    @abc.abstractmethod
    def root_slug(self):
        """
        Must return a pathlike unique identifier for the root of the storage object.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def recognizes(root, comm):
        """
        Must return whether the given root argument is recognized as a valid storage object.

        :param root: The unique identifier for the storage
        :param mpi4py.MPI.Comm comm: MPI communicator that shares control
          over the Storage.
        """
        pass

    @classmethod
    def peek_exists(cls, root):
        """
        Must peek at the existence of the given root, without instantiating anything.
        """
        try:
            return Path(root).exists()
        except Exception:
            return False

    @abc.abstractmethod
    def exists(self):
        """
        Must check existence of the storage object.
        """
        pass

    @abc.abstractmethod
    def create(self):
        """
        :guilabel:`collective` Must create the storage engine.
        """
        pass

    @abc.abstractmethod
    def move(self, new_root):
        """
        :guilabel:`collective` Must move the storage object to the new root.
        """
        pass

    @abc.abstractmethod
    def copy(self, new_root):
        """
        :guilabel:`collective` Must copy the storage object to the new root.
        """
        pass

    @abc.abstractmethod
    def remove(self):
        """
        :guilabel:`collective` Must remove the storage object.
        """
        pass

    @abc.abstractmethod
    def clear_placement(self):
        """
        :guilabel:`collective` Must clear existing placement data.
        """
        pass

    @abc.abstractmethod
    def clear_connectivity(self):
        """
        :guilabel:`collective` Must clear existing connectivity data.
        """
        pass

    @abc.abstractmethod
    def get_chunk_stats(self):
        """
        :guilabel:`readonly` Must return a dictionary with all chunk statistics.
        """
        pass

    def read_only(self):
        """
        A context manager that enters the engine into readonly mode. In
        readonly mode the engine does not perform any locking, write-operations or network
        synchronization, and errors out if a write operation is attempted.
        """
        self._readonly = True
        return ReadOnlyManager(self)

    def readwrite(self):
        self._readonly = False


class ReadOnlyManager:
    def __init__(self, engine):
        self._e = engine

    def __enter__(self):
        self._e._readonly = True

    def __exit__(self, *args):
        self._e._readonly = False


class NetworkDescription(Interface):
    pass


class FileStore(Interface, engine_key="files"):
    """
    Interface for the storage and retrieval of files essential to the network description.
    """

    @abc.abstractmethod
    def all(self):
        """
        Return all ids and associated metadata in the file store.
        """
        pass

    @abc.abstractmethod
    def store(self, content, id=None, meta=None, encoding=None, overwrite=False):
        """
        Store content in the file store. Should also store the current timestamp as
        `mtime` meta.

        :param content: Content to be stored
        :type content: str
        :param id: Optional specific id for the content to be stored under.
        :type id: str
        :param meta: Metadata for the content
        :type meta: dict
        :param encoding: Optional encoding
        :type encoding: str
        :param overwrite: Overwrite existing file
        :type overwrite: bool
        :returns: The id the content was stored under
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def load(self, id):
        """
        Load the content of an object in the file store.

        :param id: id of the content to be loaded.
        :type id: str
        :returns: The content of the stored object
        :rtype: str
        :raises FileNotFoundError: The given id doesn't exist in the file store.
        """
        pass

    @abc.abstractmethod
    def remove(self, id):
        """
        Remove the content of an object in the file store.

        :param id: id of the content to be removed.
        :type id: str
        :raises FileNotFoundError: The given id doesn't exist in the file store.
        """
        pass

    @abc.abstractmethod
    def store_active_config(self, config):
        """
        Store configuration in the file store and mark it as the active configuration of
        the stored network.

        :param config: Configuration to be stored
        :type config: :class:`~.config.Configuration`
        :returns: The id the config was stored under
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def load_active_config(self):
        """
        Load the active configuration stored in the file store.

        :returns: The active configuration
        :rtype: :class:`~.config.Configuration`
        :raises Exception: When there's no active configuration in the file store.
        """
        pass

    @abc.abstractmethod
    def has(self, id):
        """
        Must return whether the file store has a file with the given id.
        """
        pass

    @abc.abstractmethod
    def get_mtime(self, id):
        """
        Must return the last modified timestamp of file with the given id.
        """
        pass

    @abc.abstractmethod
    def get_encoding(self, id):
        """
        Must return the encoding of the file with the given id, or None if it is
        unspecified binary data.
        """
        pass

    @abc.abstractmethod
    def get_meta(self, id) -> typing.Mapping[str, typing.Any]:
        """
        Must return the metadata of the given id.
        """
        pass

    def get(self, id) -> "StoredFile":
        """
        Return a StoredFile wrapper
        """
        if not self.has(id):
            raise FileNotFoundError(f"File with id '{id}' not found.")
        return StoredFile(self, id)

    def find_files(self, predicate):
        return (
            StoredFile(self, id_) for id_, m in self.all().items() if predicate(id_, m)
        )

    def find_file(self, predicate):
        return next(self.find_files(predicate), None)

    def find_id(self, id):
        return self.find_file(lambda id_, _: id_ == id)

    def find_meta(self, key, value):
        return self.find_file(lambda _, meta: meta.get(key, None) == value)


class StoredFile:
    def __init__(self, store, id):
        self.store = store
        self.id = id

    @property
    def meta(self):
        return self.store.get_meta(self.id)

    @property
    def mtime(self):
        return self.store.get_mtime(self.id)

    def load(self):
        return self.store.load(self.id)


class PlacementSet(Interface):
    """
    Interface for the storage of placement data of a cell type.
    """

    @abc.abstractmethod
    def __init__(self, engine, cell_type):
        super().__init__(engine)
        self._type = cell_type
        self._tag = cell_type.name
        self._morphology_labels = None

    @abc.abstractmethod
    def __len__(self):
        pass

    @obj_str_insert
    def __repr__(self):
        cell_type = self.cell_type
        try:
            ms = self.load_morphologies()
        except Exception:
            return f"cell type: '{cell_type.name}'"
        if not len(ms):
            mstr = "without morphologies"
        else:
            mstr = f"with {len(ms._loaders)} morphologies"
        return f"cell type: '{cell_type.name}', {len(self)} cells, {mstr}"

    @property
    def cell_type(self):
        """
        The associated cell type.

        :returns: The cell type
        :rtype: ~bsb.cell_types.CellType
        """
        return self._type

    @property
    def tag(self):
        """
        The unique identifier of the placement set.

        :returns: Unique identifier
        :rtype: str
        """
        return self._tag

    @classmethod
    @abc.abstractmethod
    def create(cls, engine, cell_type):
        """
        Create a placement set.

        :param engine: The engine that governs this PlacementSet.
        :type engine: `bsb.storage.interfaces.Engine`
        :param cell_type: The cell type whose data is stored in the placement set.
        :type cell_type: bsb.cell_types.CellType
        :returns: A placement set
        :rtype: bsb.storage.interfaces.PlacementSet
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def exists(engine, cell_type):
        """
        Check existence of a placement set.

        :param engine: The engine that governs the existence check.
        :type engine: `bsb.storage.interfaces.Engine`
        :param cell_type: The cell type to look for.
        :type cell_type: bsb.cell_types.CellType
        :returns: Whether the placement set exists.
        :rtype: bool
        """
        pass

    @classmethod
    def require(cls, engine, cell_type):
        """
        Return and create a placement set, if it didn't exist before.

        The default implementation uses the
        :meth:`~bsb.storage.interfaces.PlacementSet.exists` and
        :meth:`~bsb.storage.interfaces.PlacementSet.create` methods.

        :param engine: The engine that governs this PlacementSet.
        :type engine: `bsb.storage.interfaces.Engine`
        :param cell_type: The cell type whose data is stored in the placement set.
        :type cell_type: bsb.cell_types.CellType
        :returns: A placement set
        :rtype: bsb.storage.interfaces.PlacementSet
        """
        if not cls.exists(engine, cell_type):
            cls.create(engine, cell_type)
        return cls(engine, cell_type)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Clear (some chunks of) the placement set.

        :param chunks: If given, the specific chunks to clear.
        :type chunks: List[bsb.storage._chunks.Chunk]
        """
        pass

    @abc.abstractmethod
    def get_all_chunks(self):
        """
        Get all the chunks that exist in the placement set.

        :returns: List of existing chunks.
        :rtype: List[bsb.storage._chunks.Chunk]
        """
        pass

    @abc.abstractmethod
    def load_ids(self):
        pass

    @abc.abstractmethod
    def load_positions(self):
        """
        Return a dataset of cell positions.

        :returns: An (Nx3) dataset of positions.
        :rtype: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def load_rotations(self):
        """
        Load the rotation data of the placement set
        :returns: A rotation set
        :rtype: ~bsb.morphologies.RotationSet
        """
        pass

    @abc.abstractmethod
    def load_morphologies(self, allow_empty=False):
        """
        Return a :class:`~.morphologies.MorphologySet` associated to the cells. Raises an
        error if there is no morphology data, unless `allow_empty=True`.

        :param bool allow_empty: Silence missing morphology data error, and return an
          empty morphology set.
        :returns: Set of morphologies
        :rtype: :class:`~.morphologies.MorphologySet`
        """
        pass

    @abc.abstractmethod
    def load_additional(self, key=None):
        pass

    def count_morphologies(self):
        """
        Must return the number of different morphologies used in the set.
        """
        return self.load_morphologies(allow_empty=True).count_morphologies()

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def append_data(
        self,
        chunk,
        positions=None,
        morphologies=None,
        rotations=None,
        additional=None,
        count=None,
    ):
        """
        Append data to the placement set. If any of ``positions``, ``morphologies``, or
        ``rotations`` is given, the arguments to its left must also be given (e.g. passing
        morphologies, but no positions, is not allowed, passing just positions is allowed)

        :param chunk: The chunk to store data in.
        :type chunk: ~bsb.storage._chunks.Chunk
        :param positions: Cell positions
        :type positions: numpy.ndarray
        :param rotations: Cell rotations
        :type rotations: ~bsb.morphologies.RotationSet
        :param morphologies: Cell morphologies
        :type morphologies: ~bsb.morphologies.MorphologySet
        :param additional: Additional datasets with 1 value per cell, will be stored
          under its key in the dictionary
        :type additional: Dict[str, numpy.ndarray]
        :param count: Amount of entities to place. Excludes the use of any positional,
          rotational or morphological data.
        :type count: int
        """
        pass

    @abc.abstractmethod
    def append_additional(self, name, chunk, data):
        """
        Append arbitrary user data to the placement set. The length of the data must match
        that of the placement set, and must be storable by the engine.

        :param name:
        :param chunk: The chunk to store data in.
        :type chunk: ~bsb.storage._chunks.Chunk
        :param data: Arbitrary user data. You decide |:heart:|
        :type data: numpy.ndarray
        """
        pass

    @abc.abstractmethod
    def chunk_context(self, chunks):
        pass

    @abc.abstractmethod
    def set_chunk_filter(self, chunks):
        """
        Should limit the scope of the placement set to the given chunks.

        :param chunks: List of chunks
        :type chunks: list[bsb.storage._chunks.Chunk]
        """
        pass

    @abc.abstractmethod
    def set_label_filter(self, labels):
        """
        Should limit the scope of the placement set to the given labels.

        :param labels: List of labels
        :type labels: list[str]
        """
        pass

    @abc.abstractmethod
    def set_morphology_label_filter(self, morphology_labels):
        """
        Should limit the scope of the placement set to the given sub-cellular labels. The
        morphologies returned by
        :meth:`~.storage.interfaces.PlacementSet.load_morphologies` should return a
        filtered form of themselves if :meth:`~.morphologies.Morphology.as_filtered` is
        called on them.

        :param morphology_labels: List of labels
        :type morphology_labels: list[str]
        """
        pass

    @abc.abstractmethod
    def label(self, labels, cells):
        """
        Should label the cells with given labels.

        :param cells: Array of cells in this set to label.
        :type cells: numpy.ndarray
        :param labels: List of labels
        :type labels: list[str]
        """
        pass

    @abc.abstractmethod
    def get_labelled(self, labels):
        """
        Should return the ids of the cells labelled with given labels.

        :param labels: List of labels
        :type labels: list[str]
        """
        pass

    @abc.abstractmethod
    def get_label_mask(self, labels):
        """
        Should return a mask that fits the placement set for the cells with given labels.

        :param labels: List of labels
        :type labels: list[str]
        """
        pass

    @abc.abstractmethod
    def get_chunk_stats(self):
        """
        Should return how many cells were placed in each chunk.
        """
        pass

    def load_boxes(self, morpho_cache=None):
        """
        Load the cells as axis aligned bounding box rhomboids matching the extension,
        orientation and position in space. This function loads morphologies, unless a
        `morpho_cache` is given, then that is used.

        :param morpho_cache: If you've previously loaded morphologies with soft or hard
          caching enabled, you can pass the resulting morphology set here to reuse it. If
          afterwards you need the morphology set, you best call :meth:`.load_morphologies`
          first and reuse it here.
        :type morpho_cache: ~bsb.morphologies.MorphologySet
        :returns: An iterator with 6 coordinates per cell: 3 min and 3 max coords, the
          bounding box of that cell's translated and rotated morphology.
        :rtype: Iterator[Tuple[float, float, float, float, float, float]]
        :raises: DatasetNotFoundError if no morphologies are found.
        """
        if morpho_cache is None:
            mset = self.load_morphologies()
        else:
            mset = morpho_cache
        expansion = [*zip([0] * 4 + [1] * 4, ([0] * 2 + [1] * 2) * 2, [0, 1] * 4)]

        def _box_of(m, o, r):
            oo = (m["ldc"], m["mdc"])
            # Make the 8 corners of the box
            corners = np.array([[oo[x][0], oo[y][1], oo[z][2]] for x, y, z in expansion])
            # Rotate them
            rotbox = r.apply(corners)
            # Find outer box, by rotating and translating the starting box
            return np.concatenate(
                (np.min(rotbox, axis=0) + o, np.max(rotbox, axis=0) + o)
            )

        iters = (mset.iter_meta(), self.load_positions(), self.load_rotations())
        return map(_box_of, *iters)

    def load_box_tree(self, morpho_cache=None):
        """
        Load boxes, and form an RTree with them, for fast spatial lookup of rhomboid
        intersection.

        :param morpho_cache: See :meth:`~bsb.storage.interfaces.PlacementSet.load_boxes`.
        :returns: A boxtree
        :rtype: bsb.trees.BoxTree
        """
        return BoxTree(list(self.load_boxes(morpho_cache=morpho_cache)))

    def _requires_morpho_mapping(self):
        return self._morphology_labels is not None and self.count_morphologies()

    def _morpho_backmap(self, locs):
        locs = locs.copy()
        cols = locs[:, 1:]
        ign_b = cols[:, 0] == -1
        ign_p = cols[:, 1] == -1
        semi = ign_b != ign_p
        if np.any(semi):
            raise ValueError(
                f"Invalid data at {np.nonzero(semi)[0]}. -1 needs to occur in "
                "either none or both columns to make point neuron connections."
            )
        to_map = ~ign_b
        if np.any(locs[to_map, 1:] < 0):
            raise ValueError(
                f"Invalid data at {np.nonzero(locs[to_map, 1:] < 0)[0]}, "
                "negative values are not valid morphology locations."
            )
        locs[to_map] = self.load_morphologies()._mapback(locs[to_map])
        return locs


class MorphologyRepository(Interface, engine_key="morphologies"):
    @abc.abstractmethod
    def all(self):
        """
        Fetch all the stored morphologies.

        :returns: List of the stored morphologies.
        :rtype: List[~bsb.storage.interfaces.StoredMorphology]
        """
        pass

    @abc.abstractmethod
    def select(self, *selectors):
        """
        Select stored morphologies.

        :param selectors: Any number of morphology selectors.
        :type selectors: List[bsb.morphologies.selector.MorphologySelector]
        :returns: All stored morphologies that match at least one selector.
        :rtype: List[~bsb.storage.interfaces.StoredMorphology]
        """
        pass

    @abc.abstractmethod
    def save(self, name, morphology, overwrite=False):
        """
        Store a morphology

        :param name: Key to store the morphology under.
        :type name: str
        :param morphology: Morphology to store
        :type morphology: bsb.morphologies.Morphology
        :param overwrite: Overwrite any stored morphology that already exists under that
          name
        :type overwrite: bool
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
        pass

    @abc.abstractmethod
    def has(self, name):
        """
        Check whether a morphology under the given name exists

        :param name: Key of the stored morphology.
        :type name: str
        :returns: Whether the key exists in the repo.
        :rtype: bool
        """
        pass

    def __contains__(self, item):
        return self.has(item)

    @abc.abstractmethod
    def preload(self, name):
        """
        Load a stored morphology as a morphology loader.

        :param name: Key of the stored morphology.
        :type name: str
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
        pass

    @abc.abstractmethod
    def load(self, name):
        """
        Load a stored morphology as a constructed morphology object.

        :param name: Key of the stored morphology.
        :type name: str
        :returns: A morphology
        :rtype: ~bsb.morphologies.Morphology
        """
        pass

    @abc.abstractmethod
    def get_meta(self, name):
        """
        Get the metadata of a stored morphology.

        :param name: Key of the stored morphology.
        :type name: str
        :returns: Metadata dictionary
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def get_all_meta(self):
        """
        Get the metadata of all stored morphologies.
        :returns: Metadata dictionary
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def set_all_meta(self, all_meta):
        """
        Set the metadata of all stored morphologies.
        :param all_meta: Metadata dictionary.
        :type all_meta: dict
        """
        pass

    @abc.abstractmethod
    def update_all_meta(self, meta):
        """
        Update the metadata of stored morphologies with the provided key values

        :param meta: Metadata dictionary.
        :type meta: str
        """
        pass

    def list(self):
        """
        List all the names of the morphologies in the repository.
        """
        return [loader.name for loader in self.all()]


class ConnectivitySet(Interface):
    """
    Stores the connections between 2 types of cell as ``local`` and ``global`` locations.
    A location is a cell id, referring to the n-th cell in the chunk, a branch id, and a
    point id, to specify the location on the morphology. Local locations refer to cells on
    this chunk, while global locations can come from any chunk and is associated to a
    certain chunk id as well.

    Locations are either placement-context or chunk dependent: You may form connections
    between the n-th cells of a placement set (using
    :meth:`~.storage.interfaces.ConnectivitySet.connect`), or of the n-th cells of 2
    chunks (using :meth:`~.storage.interfaces.ConnectivitySet.chunk_connect`).

    A cell has both incoming and outgoing connections; when speaking of incoming
    connections, the local locations are the postsynaptic cells, and when speaking of
    outgoing connections they are the presynaptic cells. Vice versa for the global
    connections.
    """

    # The following attributes must be set on each ConnectivitySet by the engine:
    tag: str
    pre_type_name: str
    post_type_name: str
    pre_type: "CellType"
    post_type: "CellType"

    @abc.abstractmethod
    def __len__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def create(cls, engine, tag):
        """
        Must create the placement set.
        """
        pass

    @obj_str_insert
    def __repr__(self):
        cstr = f"with {len(self)} connections" if len(self) else "without connections"
        return f"'{self.tag}' {cstr}"

    @staticmethod
    @abc.abstractmethod
    def exists(engine, tag):
        """
        Must check the existence of the connectivity set
        """
        pass

    def require(self, engine, tag):
        """
        Must make sure the connectivity set exists. The default
        implementation uses the class's ``exists`` and ``create`` methods.
        """
        if not self.exists(engine, tag):
            self.create(engine, tag)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Must clear (some chunks of) the placement set
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_tags(cls, engine):
        """
        Must return the tags of all existing connectivity sets.

        :param engine: Storage engine to inspect.
        """
        pass

    @abc.abstractmethod
    def connect(self, pre_set, post_set, src_locs, dest_locs):
        """
        Must connect the ``src_locs`` to the ``dest_locs``, interpreting the cell ids
        (first column of the locs) as the cell rank in the placement set.
        """
        pass

    @abc.abstractmethod
    def chunk_connect(self, src_chunk, dst_chunk, src_locs, dst_locs):
        """
        Must connect the ``src_locs`` to the ``dest_locs``, interpreting the cell ids
        (first column of the locs) as the cell rank in the chunk.
        """
        pass

    @abc.abstractmethod
    def get_local_chunks(self, direction):
        """
        Must list all the local chunks that contain data in the given ``direction``
        (``"inc"`` or ``"out"``).
        """
        pass

    @abc.abstractmethod
    def get_global_chunks(self, direction, local_):
        """
        Must list all the global chunks that contain data coming from a ``local`` chunk
        in the given ``direction``
        """
        pass

    @abc.abstractmethod
    def nested_iter_connections(self, direction=None, local_=None, global_=None):
        """
        Must iterate over the connectivity data, leaving room for the end-user to set up
        nested for loops:

        .. code-block:: python

          for dir, itr in self.nested_iter_connections():
              for lchunk, itr in itr:
                  for gchunk, data in itr:
                      print(f"Nested {dir} block between {lchunk} and {gchunk}")

        If a keyword argument is given, that axis is not iterated over, and the amount of
        nested loops is reduced.
        """
        pass

    @abc.abstractmethod
    def flat_iter_connections(self, direction=None, local_=None, global_=None):
        """
        Must iterate over the connectivity data, yielding the direction, local chunk,
        global chunk, and data:

        .. code-block:: python

          for dir, lchunk, gchunk, data in self.flat_iter_connections():
              print(f"Flat {dir} block between {lchunk} and {gchunk}")

        If a keyword argument is given, that axis is not iterated over, and the value is
        fixed in each iteration.
        """
        pass

    @abc.abstractmethod
    def load_block_connections(self, direction, local_, global_):
        """
        Must load the connections from ``direction`` perspective between ``local_`` and
        ``global_``.

        :returns: The local and global connections locations
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        pass

    @abc.abstractmethod
    def load_local_connections(self, direction, local_):
        """
        Must load all the connections from ``direction`` perspective in ``local_``.

        :returns: The local connection locations, a vector of the global connection chunks
          (1 chunk id per connection), and the global connections locations. To identify a
          cell in the global connections, use the corresponding chunk id from the second
          return value.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        pass

    def load_connections(self):
        """
        Loads connections as a ``CSIterator``.

        :returns: A connectivity set iterator, that will load data
        """
        return ConnectivityIterator(self, "out")


class ConnectivityIterator:
    def __init__(
        self, cs: ConnectivitySet, direction, lchunks=None, gchunks=None, scoped=True
    ):
        self._cs = cs
        self._dir = direction
        self._lchunks = lchunks
        self._scoped = scoped
        self._gchunks = gchunks

    def __copy__(self):
        lchunks = self._lchunks.copy() if self._lchunks is not None else None
        gchunks = self._gchunks.copy() if self._gchunks is not None else None
        return ConnectivityIterator(self._cs, self._dir, lchunks, gchunks)

    def __len__(self):
        return len(self.all()[0])

    def __iter__(self):
        """
        Iterate over the connection locations chunk by chunk.

        :returns: The presynaptic location matrix and postsynaptic location matrix.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        for _, pre_locs, _, post_locs in self.chunk_iter():
            yield from zip(pre_locs, post_locs)

    def chunk_iter(self):
        """
        Iterate over the connection data chunk by chunk.

        :returns: The presynaptic chunk, presynaptic locations, postsynaptic chunk,
          and postsynaptic locations.
        :rtype: Tuple[~bsb.storage._chunks.Chunk, numpy.ndarray, ~bsb.storage._chunks.Chunk, numpy.ndarray]
        """
        yield from (
            self._offset_block(*data)
            for data in self._cs.flat_iter_connections(
                self._dir, self._lchunks, self._gchunks
            )
        )

    @immutable()
    def as_globals(self):
        self._scoped = False

    @immutable()
    def as_scoped(self):
        self._scoped = True

    @immutable()
    def outgoing(self):
        self._dir = "out"
        self._lchunks, self._gchunks = self._gchunks, self._lchunks

    @immutable()
    def incoming(self):
        self._dir = "inc"
        self._lchunks, self._gchunks = self._gchunks, self._lchunks

    @immutable()
    def to(self, chunks):
        if isinstance(chunks, Chunk) and chunks.ndim == 1:
            chunks = [chunks]
        if self._dir == "inc":
            self._lchunks = chunks
        else:
            self._gchunks = chunks

    @immutable()
    def from_(self, chunks):
        if isinstance(chunks, Chunk) and chunks.ndim == 1:
            chunks = [chunks]
        if self._dir == "out":
            self._lchunks = chunks
        else:
            self._gchunks = chunks

    def all(self):
        pre_blocks = []
        post_blocks = []
        lens = []
        for _, pre_block, _, post_block in self.chunk_iter():
            pre_blocks.append(pre_block)
            post_blocks.append(post_block)
            lens.append(len(pre_block))
        pre_locs = np.empty((sum(lens), 3), dtype=int)
        post_locs = np.empty((sum(lens), 3), dtype=int)
        ptr = 0
        for len_, pre_block, post_block in zip(lens, pre_blocks, post_blocks):
            pre_locs[ptr : ptr + len_] = pre_block
            post_locs[ptr : ptr + len_] = post_block
            ptr += len_
        return pre_locs, post_locs

    def _offset_block(self, direction: str, lchunk, gchunk, data):
        loff = self._local_chunk_offsets()
        goff = self._global_chunk_offsets()
        llocs, glocs = data
        llocs[:, 0] += loff[lchunk]
        glocs[:, 0] += goff[gchunk]
        if direction == "out":
            return lchunk, llocs, gchunk, glocs
        else:
            return gchunk, glocs, lchunk, llocs

    @functools.cache
    def _local_chunk_offsets(self):
        source = self._cs.post_type if self._dir == "inc" else self._cs.pre_type
        return self._chunk_offsets(source, self._lchunks)

    @functools.cache
    def _global_chunk_offsets(self):
        source = self._cs.pre_type if self._dir == "inc" else self._cs.post_type
        return self._chunk_offsets(source, self._gchunks)

    def _chunk_offsets(self, source, chunks):
        stats = source.get_placement_set().get_chunk_stats()
        if self._scoped and chunks is not None:
            stats = {chunk: item for chunk, item in stats.items() if int(chunk) in chunks}
        offsets = {}
        ctr = 0
        for chunk, len_ in sorted(
            stats.items(), key=lambda k: Chunk.from_id(int(k[0]), None).id
        ):
            offsets[Chunk.from_id(int(chunk), None)] = ctr
            ctr += len_
        return offsets


class StoredMorphology:
    def __init__(self, name, loader, meta):
        self.name = name
        self._loader = loader
        self._meta = meta

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def get_meta(self):
        return self._meta.copy()

    def load(self):
        return self._loader()

    def cached_load(self, labels=None):
        if labels is not None:
            labels = tuple(labels)
        return self._cached_load(labels)

    @functools.cache
    def _cached_load(self, labels):
        return self.load().set_label_filter(labels).as_filtered()


class GeneratedMorphology(StoredMorphology):
    def __init__(self, name, generated, meta):
        super().__init__(name, lambda: generated, meta)


__all__ = [
    "ConnectivityIterator",
    "ConnectivitySet",
    "Engine",
    "FileStore",
    "GeneratedMorphology",
    "Interface",
    "MorphologyRepository",
    "NetworkDescription",
    "NoopLock",
    "PlacementSet",
    "ReadOnlyManager",
    "StorageNode",
    "StoredFile",
    "StoredMorphology",
]
