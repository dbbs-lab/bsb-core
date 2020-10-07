import pkg_resources, errr
from .exceptions import *
import types


def discover(category, *args, **kwargs):
    registry = {}
    for entry in pkg_resources.iter_entry_points("bsb." + category):
        try:
            advert = entry.load()
            if hasattr(advert, "__plugin__"):
                advert = advert.__plugin__
            # Use `types.FunctionType` over `callable` as `callable` might confuse plugin
            # objects that have a `__call__` method with plugin factory functions.
            if isinstance(advert, types.FunctionType):
                advert = advert()
            registry[name] = advert
            _decorate_advert(advert, entry)
        except Exception as e:  # pragma: nocover
            errr.wrap(
                PluginError,
                e,
                plugin,
                prepend="Could not instantiate the `%plugin.name%` plugin:\n",
            )

    return registry

def _decorate_advert(advert, entry):
    advert._bsb_entry_point = entry
