import abc
import types
import functools
from contextlib import contextmanager
import numpy as np
from ..morphologies import Morphology, Branch
from ..trees import BoxTree
from rtree import index as rtree
from scipy.spatial.transform import Rotation


class Interface(abc.ABC):
    _iface_engine_key = None

    def __init__(self, engine):
        self._engine = engine

    def __init_subclass__(cls, **kwargs):
        # Only change engine key if explicitly given.
        if "engine_key" in kwargs:
            cls._iface_engine_key = kwargs["engine_key"]


class Engine(Interface):
    def __init__(self, root):
        self.root = root

    @property
    def format(self):
        # This attribute is set on the engine by the storage provider and correlates to
        # the name of the engine plugin.
        return self._format

    @abc.abstractmethod
    def exists(self):
        pass

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def move(self, new_root):
        pass

    @abc.abstractmethod
    def remove(self):
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
    def exists(self, engine, cell_type):
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
        :type selectors: List[bsb.placement.indicator.MorphologySelector]
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

    def import_swc(self, file, name, overwrite=False):
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
        morpho = Morphology.from_swc(file)

        return self.save(name, morpho, overwrite=overwrite)

    def import_asc(self, file, name, overwrite=False):
        """
        Import and store .asc file contents as a morphology in the repository.

        :param file: file-like object or path to the file.
        :param name: Key to store the morphology under.
        :type name: str
        :param overwrite: Overwrite any stored morphology that already exists under that
          name
        :type overwrite: bool
        :returns: The stored morphology
        :rtype: ~bsb.storage.interfaces.StoredMorphology
        """
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


class ConnectivitySet(Interface):
    @abc.abstractclassmethod
    def create(cls, engine, tag):
        """
        Override with a method to create the placement set.
        """
        pass

    @abc.abstractstaticmethod
    def exists(self, engine, tag):
        """
        Override with a method to check existence of the placement set
        """
        pass

    def require(self, engine, tag):
        """
        Can be overridden with a method to make sure the placement set exists. The default
        implementation uses the class's ``exists`` and ``create`` methods.
        """
        if not self.exists(engine, tag):
            self.create(engine, tag)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Override with a method to clear (some chunks of) the placement set
        """
        pass

    @abc.abstractclassmethod
    def get_tags(cls, engine):
        pass


class StoredMorphology:
    def __init__(self, name, loader, meta):
        self.name = name
        self._loader = loader
        self._meta = meta

    def get_meta(self):
        return self._meta.copy()

    def load(self):
        return self._loader()

    @functools.cache
    def cached_load(self):
        return self.load()
