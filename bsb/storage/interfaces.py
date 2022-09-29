import abc
from pathlib import Path
import functools
import numpy as np
from ..morphologies import Morphology
from ..trees import BoxTree
from .._util import obj_str_insert


class Interface(abc.ABC):
    _iface_engine_key = None

    def __init__(self, engine):
        self._engine = engine

    def __init_subclass__(cls, **kwargs):
        # Only change engine key if explicitly given.
        if "engine_key" in kwargs:
            cls._iface_engine_key = kwargs["engine_key"]


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
        Set a new communicator in charge of collective operations.
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
    def root_slug(self):
        """
        Must return a pathlike unique identifier for the root of the storage object.
        """
        pass

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
        pass

    @abc.abstractmethod
    def clear_connectivity(self):
        pass


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
    def store(self, content, id=None, meta=None):
        """
        Store content in the file store.

        :param content: Content to be stored
        :type content: str
        :param id: Optional specific id for the content to be stored under.
        :type id: str
        :param meta: Metadata for the content
        :type meta: dict
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
    def stream(self, id, binary=False):
        """
        Stream the content of an object in the file store.

        :param id: id of the content to be streamed.
        :type id: str
        :param binary: Whether to return file in text or bytes mode.
        :type binary: bool
        :returns: A readable file-like object of the content.
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


class PlacementSet(Interface):
    """
    Interface for the storage of placement data of a cell type.
    """

    @abc.abstractmethod
    def __init__(self, engine, cell_type):
        self._engine = engine
        self._type = cell_type
        self._tag = cell_type.name

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

    @abc.abstractclassmethod
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

    @abc.abstractstaticmethod
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
    def require(cls, engine, type):
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
        if not cls.exists(engine, type):
            cls.create(engine, type)
        return cls(engine, type)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Clear (some chunks of) the placement set.

        :param chunks: If given, the specific chunks to clear.
        :type chunks: List[bsb.storage.Chunk]
        """
        pass

    @abc.abstractmethod
    def get_all_chunks(self):
        """
        Get all the chunks that exist in the placement set.

        :returns: List of existing chunks.
        :rtype: List[bsb.storage.Chunk]
        """
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
    def load_morphologies(self):
        """
        Return a :class:`~.morphologies.MorphologySet` associated to the cells.

        :returns: Set of morphologies
        :rtype: :class:`~.morphologies.MorphologySet`
        """
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
    def __len__(self):
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
        :type chunk: ~bsb.storage.Chunk
        :param positions: Cell positions
        :type positions: numpy.ndarray
        :param rotations: Cell rotations
        :type rotations: ~bsb.morphologies.RotationSet
        :param morphologies: Cell morphologies
        :type morphologies: ~bsb.morphologies.MorphologySet
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

        :param chunk: The chunk to store data in.
        :type chunk: ~bsb.storage.Chunk
        :param data: Arbitrary user data. You decide |:heart:|
        :type data: numpy.ndarray
        :type count: int
        """
        pass

    @abc.abstractmethod
    def set_chunk_filter(self, chunks):
        """
        Should limit the scope of the placement set to the given chunks.

        :param chunks: List of chunks
        :type chunks: list[bsb.storage.Chunk]
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
        Should limit the scope of the placement set to the given subcellular labels. The
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
        Should return the cells labelled with given labels.

        :param cells: Array of cells in this set to label.
        :type cells: numpy.ndarray
        :param labels: List of labels
        :type labels: list[str]
        """
        pass

    @abc.abstractmethod
    def get_label_mask(self, labels):
        """
        Should return a mask that fits the placement set for the cells with given labels.

        :param cells: Array of cells in this set to label.
        :type cells: numpy.ndarray
        :param labels: List of labels
        :type labels: list[str]
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
        Fetch all of the stored morphologies.

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

    def import_swc(self, file, name=None, overwrite=False):
        """
        Import and store .swc file contents as a morphology in the repository.

        :param file: file-like object or path to the file.
        :param name: Key to store the morphology under.
        :type name: str
        :param overwrite: Overwrite any stored morphology that already exists under that
          name
        :type overwrite: bool
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
        name = name if name is not None else Path(file).stem
        morpho = Morphology.from_swc(file)

        return self.save(name, morpho, overwrite=overwrite)

    def import_file(self, file, name=None, overwrite=False):
        """
        Import and store file contents as a morphology in the repository.

        :param file: file-like object or path to the file.
        :param name: Key to store the morphology under.
        :type name: str
        :param overwrite: Overwrite any stored morphology that already exists under that
          name
        :type overwrite: bool
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
        name = name if name is not None else Path(file).stem
        morpho = Morphology.from_file(file)

        return self.save(name, morpho, overwrite=overwrite)

    def import_arb(self, arbor_morpho, labels, name, overwrite=False, centering=True):
        """
        Import and store an Arbor morphology object as a morphology in the repository.

        :param arbor_morpho: Arbor morphology.
        :type arbor_morpho: arbor.morphology
        :param name: Key to store the morphology under.
        :type name: str
        :param overwrite: Overwrite any stored morphology that already exists under that
          name
        :type overwrite: bool
        :param centering: Whether the morphology should be centered on the geometric mean
          of the morphology roots. Usually the soma.
        :type centering: bool
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
        morpho = Morphology.from_arbor(arbor_morpho, centering=centering)

        self.save(name, morpho, overwrite=overwrite)
        return morpho

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

    @abc.abstractclassmethod
    def create(cls, engine, tag):
        """
        Must create the placement set.
        """
        pass

    @obj_str_insert
    def __repr__(self):
        if not len(self):
            cstr = "without connections"
        else:
            cstr = f"with {len(self)} connections"
        return f"'{self.tag}' {cstr}"

    @abc.abstractstaticmethod
    def exists(self, engine, tag):
        """
        Must check the existence of the placement set
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

    @abc.abstractclassmethod
    def get_tags(cls, engine):
        """
        Must return the tags of all existing connectivity sets.
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

    @abc.abstractmethod
    def load_connections(self, direction="out"):
        """
        Must load all the connections from ``direction`` perspective.

        .. tip ::

            With big models, out of memory errors may occur. In which case it's better to
            use the :meth:`~.storage.interfaces.ConnectivitySet.incoming` or
            :meth:`~.storage.interfaces.ConnectivitySet.outgoing` block iterators, which
            yield the connections block by block.

        :returns: A vector of the local connection chunks (1 chunk id per connection),
          the local connection locations, a vector of the global connection chunks, and
          the global connections locations. To identify cells, match their location with
          their chunk id.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        pass

    @property
    def incoming(self):
        """
        Iterator over all the connection blocks, from the incoming perspective.
        """
        return _CSIterator(self, "inc")

    @property
    def outgoing(self):
        """
        Iterator over all the connection blocks, from the outgoing perspective.
        """
        return _CSIterator(self, "out")

    def from_(self, chunks):
        return self.outgoing.from_(chunks)

    def to(self, chunks):
        return self.outgoing.to(chunks)


class _CSIterator:
    def __init__(self, cs, dir):
        self.cs = cs
        self.dir = dir
        self.lchunks = None
        self.gchunks = None

    def __iter__(self):
        yield from self.cs.flat_iter_connections(self.dir, self.lchunks, self.gchunks)

    def to(self, chunks):
        if self.dir == "inc":
            self.lchunks = chunks
        else:
            self.gchunks = chunks
        return self

    def from_(self, chunks):
        if self.dir == "out":
            self.lchunks = chunks
        else:
            self.gchunks = chunks
        return self

    def all(self):
        lchunks = []
        gchunks = []
        locals_ = []
        globals_ = []
        for dir, lchunk, gchunk, data in self:
            lchunks.append(lchunk)
            gchunks.append(gchunk)
            locals_.append(data[0])
            globals_.append(data[1])
        lens = [len(lcl) for lcl in locals_]
        lcol = np.repeat([c.id for c in lchunks], lens)
        gcol = np.repeat([c.id for c in gchunks], lens)
        lloc = np.empty((sum(lens), 3), dtype=int)
        gloc = np.empty((sum(lens), 3), dtype=int)
        ptr = 0
        for len_, local_, global_ in zip(lens, locals_, globals_):
            lloc[ptr : ptr + len_] = local_
            gloc[ptr : ptr + len_] = global_
        return lcol, lloc, gcol, gloc


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
        self.name = name
        self._loader = lambda: generated
        self._meta = meta
