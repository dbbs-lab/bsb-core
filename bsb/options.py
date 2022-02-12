"""
This module contains the global options.

You can set options at the ``script`` level (which superceeds all other levels such as
environment variables or project settings).

.. code-block::

  import bsb.options
  from bsb.option import BsbOption

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

# Store the module magic for unpolluted namespace copy
_module_magic = globals().copy()

import sys, types
from .exceptions import OptionError, ReadOnlyOptionError
from .plugins import discover


_options = {}
_project_options = {}
_module_options = {}
_module_option_values = {}

_pre_freeze = set(globals().keys())


def _get_module_option(tag):
    global _module_options

    if tag not in _module_options:
        raise OptionError(f"Unknown module option '{tag}'")
    return _module_options[tag]


def _get_module_tag(tag):
    tags = _get_module_option(tag).__class__.script.tags
    if tags:
        return tags[0]
    else:
        return None


def get_option_classes():
    return discover("options")


def get_option(name):
    global _options

    if name in _options:
        return _options[name]
    else:
        raise OptionError(f"Unknown option '{name}'")


def register_option(name, option):
    global _options

    if name in _options:
        raise OptionError(
            f"The '{name}' option name is already taken by {_options[name].__class__}."
        )
    else:
        _options[name] = option


def register_project_option(option):
    global _project_options

    path = type(option).project.tags
    section = _project_options
    for slug in path[:-1]:
        section = section.setdefault(slug, {})

    if path[-1] in section:
        raise OptionError(
            f"The '{'.'.join(path)}' tag is already taken by {section[tag].__class__}."
        )
    else:
        section[path[-1]] = option


def get_project_option(tag):
    global _project_options
    path = tag.split(".")
    section = _project_options
    for slug in path:
        if slug in section:
            section = section[slug]
        else:
            raise OptionError(f"The project option `{tag}` does not exist.")
    return section


def register_module_option(tag, option):
    """
    Register an option as a global BSB option
    """
    global _module_options

    if tag in _module_options:
        raise OptionError(
            f"The '{tag}' tag is already taken by {_module_options[tag].__class__}."
        )
    else:
        _module_options[tag] = option


def _remove_tags(*tags):
    """
    Removes tags. Testing purposes only, undefined behavior.
    """
    global _module_options, _module_option_values
    for tag in tags:
        del _module_options[tag]


def set_module_option(tag, value):
    global _module_option_values, _module_options

    if (option := _get_module_option(tag)).readonly:
        raise ReadOnlyOptionError("'%tag%' is a read-only option.", option, tag)
    mod_tag = _get_module_tag(tag)
    if mod_tag is None:
        raise OptionError(f"'{tag}' can't be set through the options module.")
    _module_option_values[mod_tag] = value


def get_module_option(tag):
    global _module_option_values, _module_options
    tag = _get_module_tag(tag)

    if tag in _module_option_values:
        return _module_option_values[tag]
    else:
        return _module_options[tag].get()


def is_module_option_set(tag):
    global _module_option_values

    return _get_module_tag(tag) in _module_option_values


def get_options():
    global _options

    return _options.copy()


def store(path, value):
    option = get_project_option(path)
    if option is None:
        raise OptionError(f"'{path}' is not an option name.")
    option.project = value


def read(path):
    option = get_project_option(path)

    if option is None:
        raise OptionError(f"'{tag}' is not a project option.")
    return option.get(prio="project")


def get(tag, prio=None):
    option = get_option()


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


_om = _OptionsModule(__name__)
# Copy the module magic from the original module.
_om.__dict__.update(_module_magic)
# Copy over the intended API from the original module.
for _key, _value in zip(_post_freeze, map(globals().get, _post_freeze)):
    _om.__dict__[_key] = _value
# Set the module's public API.
_om.__dict__["__all__"] = sorted([k for k in vars(_om).keys() if not k.startswith("_")])

# Register the discoverable options
plugins = discover("options")
for plugin in plugins.values():
    option = plugin()
    _om.register_option(option.name, option)
    for tag in plugin.script.tags:
        _om.register_module_option(tag, option)
    if plugin.project.tags:
        _om.register_project_option(option)

sys.modules[__name__] = _om
