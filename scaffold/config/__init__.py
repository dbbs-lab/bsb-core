import sys, types

_list = list
from ._attrs import attr, list, dict, node, root, dynamic, ref, slot, pluggable
from ._make import walk_nodes
from .parsers import JsonParser

_path = __path__


class ConfigurationModule:
    def __init__(self, name):
        self.__name__ = name

    attr = staticmethod(attr)
    list = staticmethod(list)
    dict = staticmethod(dict)
    ref = staticmethod(ref)
    slot = staticmethod(slot)

    node = staticmethod(node)
    root = staticmethod(root)
    dynamic = staticmethod(dynamic)
    pluggable = staticmethod(pluggable)

    walk_nodes = staticmethod(walk_nodes)

    # The __path__ attribute needs to be retained to mark this module as a package
    __path__ = _path

    # Load the Configuration class on demand, not on import, to avoid circular
    # dependencies.
    _cfg_cls = None

    @property
    def Configuration(self):
        if self._cfg_cls is None:
            from ._config import Configuration

            self._cfg_cls = Configuration
        return self._cfg_cls

    def from_json(self, file=None, data=None):
        if file is not None:
            with open(file, "r") as f:
                data = f.read()
        tree = JsonParser(data).parse()
        return self.Configuration.__cast__(tree, None)

    __all__ = _list(vars().keys() - {"__qualname__", "__module__"})


sys.modules[__name__] = ConfigurationModule(__name__)
