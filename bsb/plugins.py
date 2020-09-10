import pkg_resources, errr
from .exceptions import *


def discover(category, *args, **kwargs):
    plugins = {}
    for plugin in pkg_resources.iter_entry_points("bsb." + category):
        try:
            advert = plugin.load()
            if hasattr(advert, "__plugin__"):
                advert = advert.__plugin__
            advert._scaffold_plugin = plugin
            plugins[plugin.name] = advert
        except Exception as e:  # pragma: nocover
            errr.wrap(
                PluginError,
                e,
                plugin,
                prepend="Could not instantiate the `%plugin.name%` plugin:\n",
            )
    return plugins
