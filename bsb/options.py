"""
This module contains the classes needed to define options and also contains the global
options.

You can set options at the ``script`` level (which superceeds all other levels such as
environment variables or project settings).

.. code-block::

  import bsb.options

  class MyOption(BsbOption, cli=("my_setting",), env=("MY_SETTING",), script=("my_setting", "my_alias")):
      def get_default(self):
          return 4

  assert bsb.options.my_setting == 4
  bsb.options.my_alias = 6
  assert bsb.options.my_setting == 6

Your ``MyOption`` will also be available on all CLI commands as ``--my_setting`` and will
be read from the ``MY_SETTING`` environment variable.
"""

import sys, types
from .exceptions import OptionError as OptionError
from .plugins import discover

_options = {}
_option_values = {}


def _tag(tag):
    global _options

    if tag not in _options:
        raise OptionError(f"Unknown option '{tag}'")
    return _options[tag].__class__.script.tags[0]


class OptionsModule(types.ModuleType):
    def __getattr__(self, attr):
        if attr == "__path__":
            # Python uses `hasattr(module, '__path__')` to see if a module is a package
            # so we need to raise an AttributeError to make `hasattr` return False.
            raise super().__getattribute__(attr)
        return self.get_module_option(attr)

    def __setattr__(self, attr, value):
        self.set_module_option(attr, value)

    def get_option_classes(self):
        return discover("options")

    def register_module_option(self, tag, option):
        global _options

        if tag in _options:
            raise OptionError(
                f"The '{tag}' tag is already taken by {_options[tag].__class__}."
            )
        else:
            _options[tag] = option

    def set_module_option(self, tag, value):
        global _option_values

        _option_values[_tag(tag)] = value

    def get_module_option(self, tag):
        global _option_values, _options
        tag = _tag(tag)

        if tag in _option_values:
            return _option_values[tag]
        else:
            return _options[tag].get()

    def is_module_option_set(self, tag):
        global _option_values

        return _tag(tag) in _option_values


om = OptionsModule(__name__)
excl = ["sys", "types", "OptionError", "OptionsModule", "discover", "om", "excl"]
vars = {k: v for k, v in globals().items() if k not in excl}
om.__dict__.update(vars)
om.__dict__["__all__"] = sorted([k for k in om.__dict__.keys() if not k.startswith("_")])

plugins = discover("options")
for plugin in plugins.values():
    option = plugin()
    for tag in plugin.script.tags:
        om.register_module_option(tag, option)

sys.modules[__name__] = om
