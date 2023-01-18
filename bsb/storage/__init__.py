"""
    This module imports all supported storage engines, objects that read and write data,
    which are present as subfolders of the `engine` folder, and provides them
    transparently to the user, as a part of the :class:`Storage <.storage.Storage>`
    factory class. The module scans the :mod:`.storage.interfaces` module for any class
    that inherits from :class:`Interface <.storage.interfaces.Interface>`  to collect all
    Feature Interfaces and then scans the ``storage.engines.*`` submodules for any class
    that provides an implementation of those features.

    These features, because they all follow the same interface can then be passed on to
    consumers and can be used independent of the underlying storage engine, which is the
    end goal of this module.
"""

from inspect import isclass
from ..exceptions import UnknownStorageEngineError
from .. import plugins
from ..services import MPI
from ._chunks import Chunk, chunklist
from ._files import FileDependency, FileDependencyNode, NrrdDependencyNode


# Pretend `Chunk` is defined here, for UX. It's only defined in `_chunks` to avoid
# circular imports anyway.
Chunk.__module__ = __name__
# Import the interfaces child module through a relative import as a sibling.
interfaces = __import__("interfaces", globals=globals(), level=1)

# Collect all classes that are a subclass of Interface except Interface itself and
# store them in a {class_name: class_object} dictionary
_storage_interfaces = {
    interface.__name__: interface
    for interface in interfaces.__dict__.values()
    if isclass(interface)
    and issubclass(interface, interfaces.Interface)
    and interface is not interfaces.Interface
}


def discover_engines():
    """
    Get a dictionary of all available storage engines.
    """
    engines = plugins.discover("storage.engines")
    for engine_name, engine_module in engines.items():
        register_engine(engine_name, engine_module)
    return engines


def get_engines():
    global _engines
    init_engines()
    return {name: module["Engine"] for name, module in _engines.items()}


def create_engine(name, root, comm):
    global _engines
    if name not in _available_engines:
        raise UnknownStorageEngineError(f"The storage engine '{name}' was not found.")
    engine = _engines[name]
    if callable(engine):
        # Initializer function found, call it to load the engine.
        _engines[name] = engine = engine()
    # Create an engine from the engine's Engine interface.
    return engine["Engine"](root, comm)


def init_engines():
    global _engines
    for engine_name, engine in _engines.items():
        if callable(engine):
            _engines[engine_name] = engine()


def register_engine(engine_name, engine_module):
    def init_engine():
        # Create engine without any supported interfaces
        engine_support = {
            interface_name: NotSupported(engine_name, interface_name)
            for interface_name in _storage_interfaces.keys()
        }
        engine_support["StorageNode"] = engine_module.StorageNode
        # Search for interface support
        for interface_name, interface in _storage_interfaces.items():
            for module_item in engine_module.__dict__.values():
                # Look through module items for child class of interface
                if (
                    isclass(module_item)
                    and module_item is not interface
                    and issubclass(module_item, interface)
                ):
                    engine_support[interface_name] = module_item
                    break

        return engine_support

    # Set the initializer as a stub for the engine. When the engine is first used, the
    # initializer is called.
    _engines[engine_name] = init_engine


class NotSupported:
    """
    Utility class that throws a ``NotSupported`` error when it is used. This is the
    default "implementation" of every storage feature that isn't provided by an engine.
    """

    _iface_engine_key = None

    def __init__(self, engine, operation):
        # Storage which engine and feature it is that isn't supported.
        self.engine = engine
        self.operation = operation

    def _unsupported_err(self):
        # Throw an error detailing the lack of support of our engine for our feature.
        raise NotImplementedError(
            "The {} storage engine does not support the {} feature".format(
                self.engine.upper(), self.operation
            )
        )

    def __call__(self, *args, **kwargs):
        self._unsupported_err()

    def __getattr__(self, attr):
        self._unsupported_err()


class Storage:
    """
    Factory class that produces all of the features and shims the functionality of the
    underlying engine.
    """

    def __init__(self, engine, root, comm=None, main=0, missing_ok=True):
        """
        Create a Storage provider based on a specific `engine` uniquely identified
        by the root object.

        :param engine: The name of the storage engine.
        :type engine: str
        :param root: An object that uniquely describes the storage, such as a filename
          or path. The value to be provided depends on the engine. For the hdf5 engine
          the filename has to be provided.
        :type root: object
        :param comm: MPI communicator that shares control over this Storage.
        :type comm: mpi4py.MPI.Comm
        :param main: Rank of the MPI process that executes single-node tasks.
        """
        self._comm = comm or MPI
        self._engine = create_engine(engine, root, self._comm)
        self._features = [
            fname for fname, supported in view_support()[engine].items() if supported
        ]
        self._engine._format = engine
        self._main = main

        # Load the engine's interface onto the object, this allows the end user to create
        # features, but it is not advised. Usually the Storage object
        # itself provides factory methods that should be used instead.
        for name, interface in _engines[engine].items():
            if name == "StorageNode":
                continue
            self.__dict__["_" + name] = interface
            # Interfaces can define an autobinding key so that singletons are available
            # on the engine under that key.
            key = interface._iface_engine_key
            if key is not None:
                if self.supports(name):
                    self._engine.__dict__[key] = interface(self._engine)
                else:
                    self._engine.__dict__[key] = NotSupported(self._engine.format, name)
        # The storage should be created at the root as soon as we initialize because
        # features might immediatly require the basic structure to be present.
        self._preexisted = self.exists()
        if not (missing_ok or self._preexisted):
            raise FileNotFoundError(f"`{engine}` storage at '{root}' does not exist.")
        if not self._preexisted:
            self.create()

    def __eq__(self, other):
        return self._engine == getattr(other, "_engine", None)

    @property
    def preexisted(self):
        return self._preexisted

    def is_main_process(self):
        return self._comm.get_rank() == self._main

    @property
    def morphologies(self):
        return self._engine.morphologies

    @property
    def files(self):
        return self._engine.files

    @property
    def root(self):
        return self._engine.root

    @property
    def root_slug(self):
        return self._engine.root_slug

    @property
    def format(self):
        return self._engine._format

    def exists(self):
        """
        Check whether the storage exists at the root.
        """
        return self._engine.exists()

    def create(self):
        """
        Create the minimal requirements at the root for other features to function and
        for the existence check to pass.
        """
        return self._engine.create()

    def copy(self, new_root):
        """
        Move the storage to a new root.
        """
        self._engine.copy(new_root)

    def move(self, new_root):
        """
        Move the storage to a new root.
        """
        self._engine.move(new_root)

    def remove(self):
        """
        Remove the storage and all data contained within. This is an irreversible
        destructive action!
        """
        self._engine.remove()

    def load(self):
        """
        Load a scaffold from the storage.

        :returns: :class:`Scaffold <.core.Scaffold>`
        """
        from ..core import Scaffold

        return Scaffold(storage=self)

    def load_active_config(self):
        """
        Load the configuration object from the storage.

        :returns: :class:`Configuration <.config.Configuration>`
        """
        return self._engine.files.load_active_config()

    def store_active_config(self, config):
        """
        Store a configuration object in the storage.
        """
        return self._engine.files.store_active_config(config)

    def supports(self, feature):
        return feature in self._features

    def assert_support(self, feature):
        if not self.supports(feature):
            raise NotImplementedError(
                "The '{}' engine lacks support for the '{}' feature.".format(
                    self._engine._format, feature
                )
            )

    def get_placement_set(self, type, chunks=None, labels=None, morphology_labels=None):
        """
        Return a PlacementSet for the given type.

        :param type: Specific cell type.
        :type type: ~bsb.cell_types.CellType
        :param chunks: Optionally load a specific list of chunks.
        :type chunks: list[tuple[float, float, float]]
        :param labels: Labels to filter the placement set by.
        :type labels: list[str]
        :returns: ~bsb.storage.interfaces.PlacementSet
        """
        ps = self._PlacementSet(self._engine, type)
        if chunks is not None:
            ps.set_chunk_filter(chunks)
        ps.set_label_filter(labels)
        ps.set_morphology_label_filter(morphology_labels)
        return ps

    def require_placement_set(self, cell_type):
        """
        Get a placement set.

        :param cell_type: Connection cell_type
        :type cell_type: ~bsb.cell_types.CellType
        :returns: ~bsb.storage.interfaces.PlacementSet
        """
        return self._PlacementSet.require(self._engine, cell_type)

    def get_connectivity_set(self, tag):
        """
        Get a connection set.

        :param tag: Connection tag
        :type tag: str
        :returns: ~bsb.storage.interfaces.ConnectivitySet
        """
        return self._ConnectivitySet(self._engine, tag)

    def require_connectivity_set(self, tag, pre=None, post=None):
        """
        Get a connection set.

        :param tag: Connection tag
        :type tag: str
        :returns: ~bsb.storage.interfaces.ConnectivitySet
        """
        return self._ConnectivitySet.require(self._engine, tag, pre, post)

    def get_connectivity_sets(self):
        """
        Return a ConnectivitySet for the given type.

        :param type: Specific cell type.
        :type type: ~bsb.cell_types.CellType
        :returns: ~bsb.storage.interfaces.ConnectivitySet
        """
        return [
            self._ConnectivitySet(self._engine, tag)
            for tag in self._ConnectivitySet.get_tags(self._engine)
        ]

    def init(self, scaffold):
        """
        Initialize the storage to be ready for use by the specified scaffold.
        """
        self.store_active_config(scaffold.configuration)
        self.init_placement(scaffold)

    def init_placement(self, scaffold):
        for cell_type in scaffold.get_cell_types():
            self.require_placement_set(cell_type)

    def renew(self, scaffold):
        """
        Remove and recreate an empty storage container for a scaffold.
        """
        self.remove()
        self.create()
        self.init(scaffold)

    def clear_placement(self, scaffold=None):
        self._engine.clear_placement()
        if scaffold is not None:
            self.init_placement(scaffold)

    def clear_connectivity(self):
        self._engine.clear_connectivity()

    def read_only(self):
        return self._engine.read_only()

    def get_chunk_stats(self):
        return self._engine.get_chunk_stats()


def open_storage(root):
    engines = get_engines()
    for name, engine in engines.items():
        if engine.peek_exists(root) and engine.recognizes(root):
            return Storage(name, root, missing_ok=False)
    else:
        for name, engine in engines.items():
            if engine.peek_exists(root):
                raise IOError(
                    f"Storage `{root}` not recognized as any installed format: "
                    + ", ".join(f"'{n}'" for n in engines.keys())
                )
        else:
            raise FileNotFoundError(f"Storage `{root}` does not exist.")


def get_engine_node(engine):
    init_engines()
    return _engines[engine]["StorageNode"]


def view_support(engine=None):
    """
    Return which storage engines support which features.
    """
    init_engines()
    if engine is None:
        return {
            # Loop over all enginges
            engine_name: {
                # Loop over all features, check whether they're supported
                feature_name: not isinstance(feature, NotSupported)
                for feature_name, feature in engine.items()
            }
            for engine_name, engine in _engines.items()
        }
    elif engine not in _engines:
        raise UnknownStorageEngineError(
            "The storage engine '{}' was not found.".format(engine)
        )
    else:
        # Loop over all features for the specific engine
        return {
            feature_name: not isinstance(feature, NotSupported)
            for feature_name, feature in _engines[engine].items()
        }


# Module setup:
# * Discover the engine plugins with `discover_engines`
# * Store engine initializers in `_engines` through `register_engine`
# Any time something from `_engines` is used, it should be checked if it is callable. If
# it is the engine plugin is not initialized yet and the function should be called and
# replaced with its return value.
_engines = {}
_available_engines = discover_engines()
