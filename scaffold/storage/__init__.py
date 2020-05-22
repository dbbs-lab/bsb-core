"""
    This module imports all supported storage engines, objects that read and write data,
    which are present as subfolders of the `engine` folder, and provides them
    transparently to the user, as a part of the :class:`Storage <.storage.Storage>`
    factory class. The module scans the :module:`.storage.interfaces` module for any class
    that inherits from :class:`Interface <.storage.interfaces.Interface>`  to collect all
    Feature Interfaces and then scans the ``storage.engines.*`` submodules for any class
    that provides an implementation of those features.

    These features, because they all follow the same interface can then be passed on to
    consumers and can be used independent of the underlying storage engine, which is the
    end goal of this module.
"""

import os
from abc import abstractmethod, ABC
from inspect import isclass
from ..exceptions import *
from ..core import Scaffold

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
# Engines directory
engine_dir = os.path.join(os.path.dirname(__file__), "engines")


def get_engines():
    """
        Get a dictionary of all available storage engines.
    """
    return {
        # Import the directory as a module and store it in the return dictionary under
        # its directory name.
        d: __import__("engines." + d, globals=globals(), level=1).__dict__[d]
        # Do this For each directory in the `engine_dir` that isn't "__pycache__"
        for d in os.listdir(engine_dir)
        if os.path.isdir(os.path.join(engine_dir, d)) and d != "__pycache__"
    }


_available_engines = get_engines()
_engines = {}


class NotSupported:
    """
        Utility class that throws a ``NotSupported`` error when it is used. This is the default
        "implementation" of every storage feature that isn't provided by an engine.
    """

    def __init__(self, engine, operation):
        # Storage which engine and feature it is that isn't supported.
        self.engine = engine
        self.operation = operation

    def __call__(self, *args, **kwargs):
        # Throw an error detailing the lack of support of our engine for our feature.
        raise NotImplementedError(
            "The {} storage engine does not support the {} feature".format(
                self.engine.upper(), self.operation
            )
        )


# Go through all available engines to determine which Interfaces are provided and therefor
# which features are supported.
for engine_name, engine_module in _available_engines.items():
    # Construct the default support dictionary where none of the features are supported
    engine_support = {
        interface_name: NotSupported(engine_name, interface_name)
        for interface_name in _storage_interfaces.keys()
    }
    # Set this engine's support to the default no support dictionary
    _engines[engine_name] = engine_support
    # Iterate over each interface that we'd like to find support for.
    for interface_name, interface in _storage_interfaces.items():
        # Iterate over all elements in the engine_module to find elements that provide
        # support for this feature
        for module_item in engine_module.__dict__.values():
            # Is it a class, not the interface itself, and a subclass of the interface?
            if (
                isclass(module_item)
                and module_item is not interface
                and issubclass(module_item, interface)
            ):
                # Then it is an implementation of the feature described by the interface.
                # Add it to the support dictionary.
                engine_support[interface_name] = module_item
                # Don't look any further through the module for this feature.
                break


class Storage:
    """
        Factory class that produces all of the features and shims the functionality of the
        underlying engine.
    """

    def __init__(self, engine, root):
        """
            Create a Storage provider based on a specific `engine` uniquely identified
            by the root object.

            :param engine: The name of the storage engine.
            :type engine: str
            :param root: An object that uniquely describes the storage, such as a filename
              or path. The value to be provided depends on the engine. For the hdf5 engine
              the filename has to be provided.
             :type root: object
        """
        if engine not in _available_engines:
            raise UnknownStorageEngineError(
                "The storage engine '{}' was not found.".format(engine)
            )
        # Load the engine's interface onto the object, this allows consumer construction
        # of features, but it is not advised. More properly the Storage object itself
        # should provide factory methods.
        for interface_name, interface in _engines[engine].items():
            self.__dict__["_" + interface_name] = interface
        # All engines should provide an Engine interface implementation, which we will use
        # to shim basic functionalities, and to pass on to features we produce.
        self._engine = self._Engine(root)
        self._root = root
        # The storage should be created at the root as soon as we initialize because
        # features might immediatly require the basic structure to be present.
        if not self.exists():
            self.create()

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
        self._engine.create()

    def move(self, new_root):
        """
            Move the storage to a new root.
        """
        self._engine.move(new_root)
        self._root = new_root

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
        config = self._Configuration(self._engine).load()
        return Scaffold(config, self)

    def get_placement_set(self, type):
        """
            Return a PlacementSet for the given type.

            :param type: Specific cell type.
            :type type: :class:`CellType <.models.CellType>`
            :returns: :class:`PlacementSet <.storage.interfaces.PlacementSet>`
        """
        return self._PlacementSet(self._engine, type)

    def init(self, scaffold):
        """
            Initialize the storage to be ready for use by the specified scaffold:

            * Create empty PlacementSets for each cell type if they're missing.

            (That's it, for now ^_^)
        """
        # Store the scaffold's configuration
        self._Configuration(self._engine).store(scaffold.configuration)
        # Make sure that at least an empty PlacementSet exists for each cell type.
        for cell_type in scaffold.get_cell_types():
            if not self._PlacementSet.exists(self._engine, cell_type):
                self._PlacementSet.create(self._engine, cell_type)

    def Label(self, label):
        """
            Factory method for the Label feature. The label feature can be used to tag
            cells with labels and to retrieve or filter by sets of labelled cells.

            :returns: :class:`Label <.storage.interfaces.Label>`
        """
        return self._Label(self._engine, label)


def view_support():
    """
        Return which storage engines support which features.
    """
    # Because sometimes it makes me feel good to write unreadable code.
    return dict(
        map(
            lambda e: (
                e[0],
                dict(
                    map(
                        lambda f: (f[0], not isinstance(f[1], NotSupported)), e[1].items()
                    )
                ),
            ),
            _engines.items(),
        )
    )
