"""
Plugins module. Uses ``pkg_resources`` to detect installed plugins and loads them as
categories.
"""


from importlib.metadata import entry_points
import errr
from .exceptions import PluginError
import types


# Before 3.10 `importlib.metadata` was provisional, and didn't have `select` yet.
class _EntryPointsPatch(dict):
    def select(self, *, group=None):
        return self.get(group, [])


def discover(category):
    """
    Discover all plugins for a given category.

    :param category: Plugin category (e.g. ``adapters`` to load all ``bsb.adapters``)
    :type category: str
    :returns: Loaded plugins by name.
    :rtype: dict
    """
    registry = {}
    eps = entry_points()
    if not hasattr(eps, "select"):
        eps = _EntryPointsPatch(eps)

    for entry in eps.select(group="bsb." + category):
        try:
            advert = entry.load()
            if hasattr(advert, "__plugin__"):
                advert = advert.__plugin__
            # Use `types.FunctionType` over `callable` as `callable` might confuse plugin
            # objects that have a `__call__` method with plugin factory functions.
            if isinstance(advert, types.FunctionType):
                advert = advert()
            registry[entry.name] = advert
            _decorate_advert(advert, entry)
        except Exception as e:  # pragma: nocover
            errr.wrap(
                PluginError,
                e,
                entry,
                prepend="Could not instantiate the `%plugin.name%` plugin:\n",
            )

    return registry


def _decorate_advert(advert, entry):
    advert._bsb_entry_point = entry
