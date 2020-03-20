from .. import __version__
from ..exceptions import *
from ..helpers import ConfigurableClass, get_qualified_class_name, suppress_stdout
from ..models import ConnectivitySet
from ..morphologies import Morphology, TrueMorphology, Compartment
from ..core import Scaffold
from abc import abstractmethod, ABC
import h5py, os, time, pickle, numpy as np
from numpy import string_
from sklearn.neighbors import KDTree
import os, sys
from importlib import import_module
from inspect import isclass

interfaces = __import__("interfaces", globals=globals(), level=1)
_storage_interfaces = {
    interface.__name__: interface
    for interface in interfaces.__dict__.values()
    if isclass(interface)
    and issubclass(interface, interfaces.Interface)
    and interface is not interfaces.Interface
}
engine_dir = os.path.join(os.path.dirname(__file__), "engines")
_available_engines = [
    d
    for d in os.listdir(engine_dir)
    if os.path.isdir(os.path.join(engine_dir, d)) and d != "__pycache__"
]
_engines = {}


class NotSupported:
    def __init__(self, engine, operation):
        self.engine = engine
        self.operation = operation

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "The {} storage enginge does not support the {} feature".format(
                self.engine.upper(), self.operation
            )
        )


for engine_name in _available_engines:
    engine_support = {
        interface_name: NotSupported(engine_name, interface_name)
        for interface_name in _storage_interfaces.keys()
    }
    _engines[engine_name] = engine_support
    engine_module = __import__(
        "engines." + engine_name, globals=globals(), level=1
    ).__dict__[engine_name]
    for module_item in engine_module.__dict__.values():
        # Check for objects in the module that inherit from a certain interface.
        for interface_name, interface in _storage_interfaces.items():
            if (
                isclass(module_item)
                and module_item is not interface
                and issubclass(module_item, interface)
            ):
                engine_support[interface_name] = module_item


class Storage:
    def __init__(self, engine, root):
        if engine not in _available_engines:
            raise UnknownStorageEngineError(
                "The storage engine '{}' was not found.".format(engine)
            )
        for interface_name, interface in _engines[engine].items():
            self.__dict__[interface_name] = interface
        self._handler = self.Engine(root)
        if not self.exists():
            self.create()

    def exists(self):
        return self._handler.exists()

    def create(self):
        self._handler.create()

    def move(self, new_root):
        self._handler.move(new_root)

    def remove(self):
        self._handler.remove()

    def load(self):
        config = self.Configuration(self._handler).load()
        return Scaffold(config, self)

    def get_placement_set(self, type):
        return self.PlacementSet(self._handler, type)

    def init(self, scaffold):
        self.Configuration(self._handler).store(scaffold.configuration)
        for cell_type in scaffold.get_cell_types():
            if not self.PlacementSet.exists(self._handler, cell_type):
                self.PlacementSet.create(self._handler, cell_type)

    def label(self, label, identifiers):
        label = self.Label(self._handler, label)
        label.add(identifiers)
        return label


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
