import pkg_resources
from .exceptions import *


def discover(category, *args, **kwargs):
    plugins = {}
    for plugin in pkg_resources.iter_entry_points("dbbs_scaffold." + category):
        try:
            advert = plugin.load()
            if hasattr(advert, "__plugin__"):
                advert = advert.__plugin__
            advert._scaffold_plugin = plugin
            plugins[plugin.name] = advert
        except:  # pragma: nocover
            raise PluginError("Could not instantiate the `{}` plugin".format(plugin.name))
    return plugins
