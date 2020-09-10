from . import attr, list, dict, node, root, pluggable, slot
from . import types
from .. import plugins


@pluggable(key="engine", plugin_name="storage engine", unpack=lambda p: p.StorageNode)
class StorageNode:
    root = slot()

    @classmethod
    def __plugins__(cls):
        if not hasattr(cls, "_plugins"):
            cls._plugins = plugins.discover("engines")
        return cls._plugins


@node
class NetworkNode:
    x = attr(type=float, required=True)
    z = attr(type=float, required=True)
