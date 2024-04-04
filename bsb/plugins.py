"""
Plugins module. Uses ``pkg_resources`` to detect installed plugins and loads them as
categories.
"""

import types
from collections import defaultdict
from importlib.metadata import entry_points
from itertools import chain

import errr

from .exceptions import PluginError


# Before 3.10 `importlib.metadata` was provisional, and didn't have `select` yet.
class _EntryPointsPatch(dict):
    def select(self, *, group=None):
        return self.get(group, [])


class mutdict(dict):
    pass


class mutlist(list):
    pass


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

    for entry in chain(eps.select(group="bsb." + category), _unittest_plugins[category]):
        try:
            advert = entry.load()
            if hasattr(advert, "__plugin__"):
                advert = advert.__plugin__
            # Use `types.FunctionType` over `callable` as `callable` might confuse plugin
            # objects that have a `__call__` method with plugin factory functions.
            if isinstance(advert, types.FunctionType):
                advert = advert()
            advert = _decorate_advert(advert, entry)
            registry[entry.name] = advert
        except Exception as e:  # pragma: nocover
            errr.wrap(
                PluginError,
                e,
                entry,
                prepend="Could not instantiate the `%plugin.name%` plugin:\n",
            )

    return registry


def _decorate_advert(advert, entry):
    if type(advert) is list:
        advert = mutlist(advert)
    elif type(advert) is dict:
        advert = mutdict(advert)
    advert._bsb_entry_point = entry
    return advert


# Registry to insert plugins without having to install them, intended for testing purposes.
_unittest_plugins = defaultdict(list)

__all__ = ["discover"]
