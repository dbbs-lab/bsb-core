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

    def __init__(self, **kwargs):
        self._distr = getattr(_distributions, self.distribution)(**self.parameters)

    def draw(self, n):
        return self._distr.rvs(size=n)

    def __getattr__(self, attr):
        if "_distr" not in self.__dict__:
            raise AttributeError("No underlying _distr found for distribution node.")
        return getattr(self._distr, attr)
