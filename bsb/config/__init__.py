"""
bsb.config module

Contains the dynamic attribute system; Use ``@bsb.config.root/node/dynamic/pluggable`` to
decorate your classes and add class attributes using ``x =
config.attr/dict/list/ref/reflist`` to populate your classes with powerful attributes.
"""

import os
import sys
import glob
import itertools
from shutil import copy2 as copy_file
import builtins
import traceback

from ._attrs import (
    attr,
    list,
    dict,
    file,
    node,
    root,
    dynamic,
    ref,
    reflist,
    slot,
    property,
    provide,
    unset,
    pluggable,
    catch_all,
    ConfigurationAttribute,
)
from .._util import ichain
from ._make import walk_node_attributes, walk_nodes
from ._hooks import on, before, after, run_hook, has_hook
from .. import plugins
from ..exceptions import ConfigTemplateNotFoundError, ParserError, PluginError


_path = __path__
ConfigurationAttribute.__module__ = __name__


class ConfigurationModule:
    from . import types, refs

    def __init__(self, name):
        self.__name__ = name

    attr = staticmethod(attr)
    list = staticmethod(list)
    dict = staticmethod(dict)
    ref = staticmethod(ref)
    reflist = staticmethod(reflist)
    slot = staticmethod(slot)
    property = staticmethod(property)
    provide = staticmethod(provide)
    catch_all = staticmethod(catch_all)
    unset = staticmethod(unset)

    node = staticmethod(node)
    root = staticmethod(root)
    dynamic = staticmethod(dynamic)
    pluggable = staticmethod(pluggable)
    file = staticmethod(file)

    walk_node_attributes = staticmethod(walk_node_attributes)
    walk_nodes = staticmethod(walk_nodes)
    on = staticmethod(on)
    after = staticmethod(after)
    before = staticmethod(before)
    run_hook = staticmethod(run_hook)
    has_hook = staticmethod(has_hook)

    _parser_classes = {}

    # The __path__ attribute needs to be retained to mark this module as a package with
    # submodules (config.refs, config.parsers.json, ...)
    __path__ = _path

    # Load the Configuration class on demand, not on import, to avoid circular
    # dependencies.
    @builtins.property
    def Configuration(self):
        if not hasattr(self, "_cfg_cls"):
            from ._config import Configuration

            self._cfg_cls = Configuration
            assert self._cfg_cls.__module__ == __name__
        return self._cfg_cls

    @builtins.property
    def ConfigurationAttribute(self):
        return ConfigurationAttribute

    def get_parser(self, parser_name):
        """
        Create an instance of a configuration parser that can parse configuration
        strings into configuration trees, or serialize trees into strings.

        Configuration trees can be cast into Configuration objects.
        """
        if parser_name not in self._parser_classes:
            raise PluginError("Configuration parser '{}' not found".format(parser_name))
        return self._parser_classes[parser_name]()

    def get_config_path(self):
        import os

        env_paths = os.environ.get("BSB_CONFIG_PATH", None)
        if env_paths is None:
            env_paths = ()
        else:
            env_paths = env_paths.split(":")
        plugin_paths = plugins.discover("config.templates")
        return [*itertools.chain((os.getcwd(),), env_paths, *plugin_paths.values())]

    def copy_template(self, template, output="network_configuration.json", path=None):
        path = [
            *map(
                os.path.abspath,
                itertools.chain(self.get_config_path(), path or ()),
            )
        ]
        for d in path:
            if files := glob.glob(os.path.join(d, template)):
                break
        else:
            raise ConfigTemplateNotFoundError(
                "'%template%' not found in config path %path%", template, path
            )
        copy_file(files[0], output)

    def from_file(self, file):
        if not hasattr(file, "read"):
            with open(file, "r") as f:
                return self.from_file(f)
        path = getattr(file, "name", None)
        if path is not None:
            path = os.path.abspath(path)
        return self.from_content(file.read(), path)

    def from_content(self, content, path=None):
        ext = path.split(".")[-1] if path is not None else None
        parser, tree, meta = _try_parsers(content, self._parser_classes, ext, path=path)
        return _from_parsed(self, parser, tree, meta, path)

    __all__ = [*(vars().keys() - {"__init__", "__qualname__", "__module__"})]


def _parser_method_docs(parser):
    mod = parser.__module__
    mod = mod[8:] if mod.startswith("scaffold.") else mod
    class_role = ":class:`{} <{}.{}>`".format(parser.__name__, mod, parser.__name__)
    if parser.data_description:
        descr = " " + parser.data_description
    else:  # pragma: nocover
        descr = ""
    return (
        "Create a Configuration object from"
        + descr
        + " data from an object or file. The data is passed to the "
        + class_role
        + """.

        :param file: Path to a file to read the data from.
        :type file: str
        :param data: Data object to hand directly to the parser
        :type data: Any
        :returns: A Configuration
        :rtype: :class:`~.config.Configuration`
    """
    )


def _try_parsers(content, classes, ext=None, path=None):  # pragma: nocover
    if ext is not None:

        def file_has_parser_ext(kv):
            return ext in getattr(kv[1], "data_extensions", ())

        classes = builtins.dict(sorted(classes.items(), key=file_has_parser_ext))
    exc = {}
    for name, cls in classes.items():
        try:
            tree, meta = cls().parse(content, path=path)
        except Exception as e:
            exc[name] = e
        else:
            return (name, tree, meta)
    msges = [
        (
            f"Can't parse contents with '{n}':\n",
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        for n, e in exc.items()
    ]
    raise ParserError("\n".join(ichain(msges)))


def _from_parsed(self, parser_name, tree, meta, file=None):
    if "components" in tree:
        from ._config import _bootstrap_components

        _bootstrap_components(tree["components"])
    conf = self.Configuration(tree)
    conf._parser = parser_name
    conf._meta = meta
    conf._file = file
    return conf


def parser_factory(name, parser):
    # This factory produces the methods for the `bsb.config.from_*` parser methods that
    # load the content of a file-like object or a simple string as a Configuration object.
    def parser_method(self, file=None, data=None, path=None):
        if file is not None:
            file = os.path.abspath(file)
            with open(file, "r") as f:
                data = f.read()
        tree, meta = parser().parse(data, path=path or file)
        return _from_parsed(self, name, tree, meta, file)

    parser_method.__name__ = "from_" + name
    parser_method.__doc__ = _parser_method_docs(parser)
    return parser_method


# Load all the `config.parsers` plugins and create a `from_*` method for them.
for name, parser in plugins.discover("config.parsers").items():
    ConfigurationModule._parser_classes[name] = parser
    setattr(ConfigurationModule, "from_" + name, parser_factory(name, parser))
    ConfigurationModule.__all__.append("from_" + name)

ConfigurationModule.__all__ = sorted(ConfigurationModule.__all__)
sys.modules[__name__] = ConfigurationModule(__name__)
