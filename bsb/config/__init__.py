"""
bsb.config module

Contains the dynamic attribute system; Use ``@bsb.config.root/node/dynamic/pluggable`` to
decorate your classes and add class attributes using ``x =
config.attr/dict/list/ref/reflist`` to populate your classes with powerful attributes.
"""

import builtins
import functools
import glob
import itertools
import os
import sys
import traceback
import typing
from shutil import copy2 as copy_file

from .. import plugins
from .._util import ichain
from ..exceptions import ConfigTemplateNotFoundError, ParserError, PluginError
from . import parsers
from ._attrs import (
    ConfigurationAttribute,
    attr,
    catch_all,
    dict,
    dynamic,
    file,
    list,
    node,
    pluggable,
    property,
    provide,
    ref,
    reflist,
    root,
    slot,
    unset,
)
from ._distributions import Distribution
from ._hooks import after, before, has_hook, on, run_hook
from ._make import (
    compose_nodes,
    get_config_attributes,
    walk_node_attributes,
    walk_nodes,
)

if typing.TYPE_CHECKING:
    from ._config import Configuration

# Add some static type hinting, to help tools figure out this dynamic module
Configuration: "Configuration"


# ConfigurationModule should not inherit from `ModuleType`, otherwise Sphinx doesn't
# document all the properties.
from . import refs, types


@functools.cache
def __getattr__(name):
    if name == "Configuration":
        # Load the Configuration class on demand, not on import, to avoid circular
        # dependencies.
        from ._config import Configuration

        return Configuration
    else:
        raise object.__getattribute__(sys.modules[__name__], name)


def get_parser(parser_name):
    """
    Create an instance of a configuration parser that can parse configuration
    strings into configuration trees, or serialize trees into strings.

    Configuration trees can be cast into Configuration objects.
    """
    parsers = plugins.discover("config.parsers")
    if parser_name not in parsers:
        raise PluginError("Configuration parser '{}' not found".format(parser_name))
    return parsers[parser_name]()


def get_config_path():
    import os

    env_paths = os.environ.get("BSB_CONFIG_PATH", None)
    if env_paths is None:
        env_paths = ()
    else:
        env_paths = env_paths.split(":")
    plugin_paths = plugins.discover("config.templates")
    return [*itertools.chain((os.getcwd(),), env_paths, *plugin_paths.values())]


def copy_configuration_template(template, output="network_configuration.json", path=None):
    path = [
        *map(
            os.path.abspath,
            itertools.chain(get_config_path(), path or ()),
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


def format_configuration_content(parser_name: str, config: "Configuration"):
    """
    Convert a configuration object to a string using the given parser.
    """
    return get_parser(parser_name).generate(config.__tree__(), pretty=True)


def make_configuration_diagram(config):
    dot = f'digraph "{config.name or "network"}" {{'
    for c in config.cell_types.values():
        dot += f'\n  {c.name}[label="{c.name}"]'
    for name, conn in config.connectivity.items():
        for pre in conn.presynaptic.cell_types:
            for post in conn.postsynaptic.cell_types:
                dot += f'\n  {pre.name} -> {post.name}[label="{name}"];'
    dot += "\n}\n"
    return dot


def _try_parsers(content, classes, ext=None, path=None):  # pragma: nocover
    if ext is not None:

        def file_has_parser_ext(kv):
            return ext not in getattr(kv[1], "data_extensions", ())

        classes = builtins.dict(sorted(classes.items(), key=file_has_parser_ext))
    exc = {}
    for name, cls in classes.items():
        try:
            tree, meta = cls().parse(content, path=path)
        except Exception as e:
            if getattr(e, "_bsbparser_show_user", False):
                raise e from None
            exc[name] = e
        else:
            return (name, tree, meta)
    msges = [
        (
            f"- Can't parse contents with '{n}':\n",
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        for n, e in exc.items()
    ]
    if path:
        msg = f"Could not parse '{path}'"
    else:
        msg = f"Could not parse content string ({len(content)} characters long)"
    raise ParserError("\n".join(ichain(msges)) + f"\n{msg}")


def _from_parsed(parser_name, tree, meta, file=None):
    conf = Configuration(tree)
    conf._parser = parser_name
    conf._meta = meta
    conf._file = file
    return conf


def parse_configuration_file(file, parser=None, path=None, **kwargs):
    if hasattr(file, "read"):
        data = file.read()
        try:
            path = str(path) or os.fspath(file)
        except TypeError:
            pass
    else:
        file = os.path.abspath(file)
        path = path or file
        with open(file, "r") as f:
            data = f.read()
    return parse_configuration_content(data, parser, path, **kwargs)


def parse_configuration_content(content, parser=None, path=None, **kwargs):
    if parser is None:
        parser_classes = plugins.discover("config.parsers")
        ext = path.split(".")[-1] if path is not None else None
        parser_name, tree, meta = _try_parsers(content, parser_classes, ext, path=path)
    elif isinstance(parser, str):
        parser_name = parser
        parser = plugins.discover("config.parsers")[parser_name](**kwargs)
        parser.parse(content, path)
        tree, meta = parser().parse(content, path=path)
    else:
        parser_name = parser.__name__
        tree, meta = parser.parse(content, path=path)
    return _from_parsed(parser_name, tree, meta, path)


# Static public API
__all__ = [
    "Configuration",
    "ConfigurationAttribute",
    "Distribution",
    "after",
    "attr",
    "before",
    "catch_all",
    "compose_nodes",
    "copy_configuration_template",
    "dict",
    "dynamic",
    "file",
    "format_configuration_content",
    "get_config_attributes",
    "get_config_path",
    "get_parser",
    "has_hook",
    "list",
    "make_configuration_diagram",
    "node",
    "on",
    "parsers",
    "parse_configuration_file",
    "parse_configuration_content",
    "pluggable",
    "property",
    "provide",
    "ref",
    "reflist",
    "root",
    "run_hook",
    "slot",
    "types",
    "unset",
    "walk_node_attributes",
    "walk_nodes",
]
__api__ = [
    "Configuration",
    "ConfigurationAttribute",
    "Distribution",
    "compose_nodes",
    "copy_configuration_template",
    "format_configuration_content",
    "get_config_attributes",
    "get_config_path",
    "get_parser",
    "make_config_diagram",
    "parsers",
    "parse_configuration_file",
    "parse_configuration_content",
    "refs",
    "types",
    "walk_node_attributes",
    "walk_nodes",
]
