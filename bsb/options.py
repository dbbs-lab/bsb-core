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
_option_values = {}

_pre_freeze = set(globals().keys())


def _get_option(tag):
    global _options

    if tag not in _options:
        raise OptionError(f"Unknown option '{tag}'")
    return _options[tag]


def _get_tag(tag):
    return _get_option(tag).__class__.script.tags[0]


def get_option_classes():
    return discover("options")


def register_module_option(tag, option):
    """
    Register an option as a global BSB option
    """
    global _options

    if tag in _options:
        raise OptionError(
            f"The '{tag}' tag is already taken by {_options[tag].__class__}."
        )
    else:
        _options[tag] = option


def _remove_tags(*tags):
    """
    Removes tags. Testing purposes only, undefined behavior.
    """
    global _options, _option_values
    for tag in tags:
        del _options[tag]


def set_module_option(tag, value):
    global _option_values, _options

    if (option := _get_option(tag)).readonly:
        raise ReadOnlyOptionError("'%tag%' is a read-only option.", option, tag)
    _option_values[_get_tag(tag)] = value


def get_module_option(tag):
    global _option_values, _options
    tag = _get_tag(tag)

    if tag in _option_values:
        return _option_values[tag]
    else:
        return _options[tag].get()


def is_module_option_set(tag):
    global _option_values

    return _get_tag(tag) in _option_values


def get_options():
    global _options

    return _options.copy()


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
    for tag in plugin.script.tags:
        _om.register_module_option(tag, option)

sys.modules[__name__] = _om
