from . import attr, list, dict, node, root, pluggable, slot, catch_all
from . import types
from .. import plugins
import scipy.stats.distributions as _distributions

_available_distributions = [
    d for d in dir(_distributions) if not d.startswith("_") and not d.endswith("_gen")
]


@pluggable(key="engine", plugin_name="storage engine")
class StorageNode:
    root = slot()

    @classmethod
    def __plugins__(cls):
        if not hasattr(cls, "_plugins"):
            cls._plugins = {
                name: plugin.StorageNode
                for name, plugin in plugins.discover("engines").items()
            }
        return cls._plugins


@node
class NetworkNode:
    x = attr(type=float, required=True)
    z = attr(type=float, required=True)


@node
class Distribution:
    distribution = attr(type=types.in_(_available_distributions), required=True)
    parameters = catch_all(type=types.any())

    def draw(self, n):
        if not hasattr(self, "_distr"):
            self._distr = getattr(_distributions, self.distribution)(**self.parameters)
        return self._distr.rvs(size=n)
