"""
This module contains the global options.

You can set options at the ``script`` level (which superceeds all other levels such as
environment variables or project settings).

.. code-block::

  import bsb.options
  from bsb import BsbOption

  class MyOption(BsbOption, cli=("my_setting",), env=("MY_SETTING",), script=("my_setting", "my_alias")):
      def get_default(self):
          return 4

  # Register the option into the `bsb.options` module
  MyOption.register()

  assert bsb.options.my_setting == 4
  bsb.options.my_alias = 6
  assert bsb.options.my_setting == 6

Your ``MyOption`` will also be available on all CLI commands as ``--my_setting`` and will
be read from the ``MY_SETTING`` environment variable.
"""

import functools

from ._options import ProfilingOption, VerbosityOption

# Store the module magic for unpolluted namespace copy
_module_magic = globals().copy()

import sys
import types

from . import option as _bsboptmod
from .exceptions import OptionError, ReadOnlyOptionError
from .plugins import discover
from .reporting import report

_options = {}
_project_options = {}
_module_options = {}
_module_option_values = {}

# Everything defined between pre-freeze and post-freeze may be considered to be added to
# the `_OptionsModule` instance, place all public API functions between them.
_pre_freeze = set(globals().keys())


def _get_module_option(tag):  # pragma: nocover
    global _module_options

    if tag not in _module_options:
        if not discover_options.cache_info().misses:
            discover_options()
            return _get_module_tag(tag)
        else:
            raise OptionError(f"Unknown module option '{tag}'")
    return _module_options[tag]


def _get_module_tag(tag):  # pragma: nocover
    return _get_module_option(tag).__class__.script.tags[0]


def get_option_classes():
    """
    Return all of the classes that are used to create singleton options from. Useful to
    access the option descriptors rather than the option values.

    :returns: The classes of all the installed options by name.
    :rtype: dict[str, bsb.option.BsbOption]
    """
    return discover("options")


def get_option_descriptor(name):
    """
    Return an option

    :param name: Name of the option to look for.
    :type name: str
    :returns: The option singleton of that name.
    :rtype: dict[str, bsb.option.BsbOption]
    """
    global _options

    discover_options()
    if name in _options:
        return _options[name]
    else:
        raise OptionError(f"Unknown option '{name}'")


def register_option(name, option):
    """
    Register an option as a global BSB option. Options that are installed by the plugin
    system are automatically registered on import of the BSB.

    :param name: Name for the option, used to store and retrieve its singleton.
    :type name: str
    :param option: Option instance, to be used as a singleton.
    :type option: :class:`.option.BsbOption`
    """
    global _options

    if name in _options:
        if type(_options[name]) != type(option):
            raise OptionError(
                f"The '{name}' option name is already taken by {_options[name].__class__}."
            )
    else:
        _options[name] = option

        for tag in type(option).script.tags:
            _register_module_option(tag, option)
        if type(option).project.tags:
            _register_project_option(option)


def unregister_option(option):
    """
    Unregister a globally registered option. Also removes its script and project parts.

    :param option: Option singleton, to be removed.
    :type option: :class:`.option.BsbOption`
    """
    global _options, _project_options

    del _options[option.name]
    _remove_module_tags(*type(option).script.tags)
    path = type(option).project.tags
    if path:
        section = _project_options
        for slug in path[:-1]:
            if slug in section:
                section = section[slug]
            else:
                return
        try:
            del section[path[-1]]
        except KeyError:
            pass


def _register_project_option(option):  # pragma: nocover
    """
    Register an option that can be manipulated from ``pyproject.toml``, unregistered
    options can be used, but :func:`.options.store` and :func:`.options.read` won't work.

    :param option: Option.
    :type option: :class:`.option.BsbOption`
    """
    global _project_options

    path = type(option).project.tags
    section = _project_options
    for slug in path[:-1]:
        section = section.setdefault(slug, {})

    if path[-1] in section:
        raise OptionError(
            f"The '{'.'.join(path)}' tag is already taken by {section[path[-1]].__class__}."
        )
    else:
        section[path[-1]] = option


def get_project_option(tag):
    """
    Find a project option

    :param tag: dot-separated path of the option. e.g. ``networks.config_link``.
    :type tag: str
    :returns: Project option instance
    :rtype: :class:`.option.BsbOption`
    """
    global _project_options
    discover_options()
    path = tag.split(".")
    section = _project_options
    for slug in path:
        if slug in section:
            section = section[slug]
        else:
            raise OptionError(f"The project option `{tag}` does not exist.")
    return section


def _register_module_option(tag, option):  # pragma: nocover
    """
    Register an option that can be manipulated from :mod:`bsb.options`.
    """
    global _module_options

    if tag in _module_options:
        raise OptionError(
            f"The '{tag}' tag is already taken by {_module_options[tag].__class__}."
        )
    else:
        _module_options[tag] = option


def _remove_module_tags(*tags):  # pragma: nocover
    """
    Removes tags.
    """
    global _module_options, _module_option_values
    for tag in tags:
        try:
            del _module_options[tag]
        except KeyError:
            pass
        try:
            del _module_option_values[tag]
        except KeyError:
            pass


def reset_module_option(tag):
    global _module_option_values
    opt = _get_module_option(tag)
    # Module option values always stored under the "module tag" (= tag 0)
    try:
        del _module_option_values[type(opt).script.tags[0]]
    except KeyError:
        pass


def set_module_option(tag, value):
    """
    Set the value of a module option. Does the same thing as ``setattr(options, tag,
    value)``.

    :param tag: Name the option is registered with in the module.
    :type tag: str
    :param value: New module value for the option
    :type value: Any
    """

    global _module_option_values

    option = _get_module_option(tag)
    if option.readonly:
        raise ReadOnlyOptionError("'%tag%' is a read-only option.", option, tag)
    mod_tag = _get_module_tag(tag)
    _module_option_values[mod_tag] = getattr(option, "setter", lambda x: x)(value)


def get_module_option(tag):
    """
    Get the value of a module option. Does the same thing as ``getattr(options, tag)``

    :param tag: Name the option is registered with in the module.
    :type tag: str
    """
    global _module_option_values, _module_options
    discover_options()
    tag = _get_module_tag(tag)

    if tag in _module_option_values:
        return _module_option_values[tag]
    else:
        return _module_options[tag].get()


def is_module_option_set(tag):
    """
    Check if a module option was set.

    :param tag: Name the option is registered with in the module.
    :type tag: str
    :returns: Whether the option was ever set from the module
    :rtype: bool
    """
    global _module_option_values

    return _get_module_tag(tag) in _module_option_values


def get_option_descriptors():
    """
    Get all the registered option singletons.
    """
    global _options

    discover_options()
    return _options.copy()


@functools.cache
def discover_options():
    # Register the discoverable options
    plugins = discover("options")
    for plugin in plugins.values():
        option = plugin()
        register_option(option.name, option)


def store_option(tag, value):
    """
    Store an option value permanently in the project settings.

    :param tag: Dot-separated path of the project option
    :type tag: str
    :param value: New value for the project option
    :type value: Any
    """
    get_project_option(tag).project = value


def read_option(tag=None):
    """
    Read an option value from the project settings. Returns all project settings if tag is
    omitted.

    :param tag: Dot-separated path of the project option
    :type tag: str
    :returns: Value for the project option
    :rtype: Any
    """
    if tag is None:
        path, content = _bsboptmod._pyproject_bsb()
        report(f"Reading project settings from '{path}'", level=4)
        return content
    else:
        return get_project_option(tag).get(prio="project")


def get_option(tag, prio=None):
    """
    Retrieve the cascaded value for an option.

    :param tag: Name the option is registered with.
    :type tag: str
    :param prio: Give priority to a type of value. Can be any of 'script', 'cli',
      'project', 'env'.
    :type prio: str
    :returns: (Possibly prioritized) value of the option.
    :rtype: Any
    """
    option = get_option_descriptor(tag)
    return option.get(prio=prio)


_post_freeze = set(globals().keys()).difference(_pre_freeze)


class _OptionsModule(types.ModuleType):
    __name__ = "bsb.options"

    def __getattr__(self, attr):
        if attr in ["__path__", "__warningregistry__", "__qualname__"]:
            # __path__:
            # Python uses `hasattr(module, '__path__')` to see if a module is a package
            # so we need to raise an AttributeError to make `hasattr` return False.
            # __warningregistry__:
            # The `unittest` module checks existence.
            # __qualname__:
            # Sphinx checks this
            raise super().__getattribute__(attr)
        return self.get_module_option(attr)

    def __setattr__(self, attr, value):
        self.set_module_option(attr, value)

    def __delattr__(self, attr):
        try:
            opt = _get_module_option(attr)
        except OptionError:
            raise super().__delattr__(attr) from None
        else:
            del opt.script


_om = _OptionsModule(__name__)
# Copy the module magic from the original module.
_om.__dict__.update(_module_magic)
# Copy over the intended API from the original module.
for _key, _value in zip(_post_freeze, map(globals().get, _post_freeze)):
    _om.__dict__[_key] = _value
# Set the module's public API.
_om.__dict__["__all__"] = sorted([k for k in vars(_om).keys() if not k.startswith("_")])

sys.modules[__name__] = _om

register_option("verbosity", VerbosityOption())
register_option("profiling", ProfilingOption())

# Static public API
__all__ = [
    "get_option",
    "get_module_option",
    "get_option_descriptor",
    "get_option_classes",
    "get_option_descriptors",
    "get_project_option",
    "is_module_option_set",
    "read_option",
    "register_option",
    "reset_module_option",
    "set_module_option",
    "store_option",
    "unregister_option",
]
