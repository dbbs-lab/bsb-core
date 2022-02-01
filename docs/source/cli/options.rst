#######
Options
#######

The command line interface uses options to specify global settings or command specific
settings. Options can be set using environment variables, the CLI command and/or through
the ``bsb.options`` module, listed here in ascending order of priority. Custom options can
be registered through the ``bsb.options`` entry point or directly using either
:func:`.options.register_module_option` or :func:`.option.BsbOption.register`.

.. note::

  Add your custom options to the ``bsb.options`` entry point to have it detected by CLI
  commands.

========================
Writing your own options
========================

You can create your own options by defining a class that inherits from :class:

.. code-block:: python

  from bsb.options import BsbOption
  from bsb.reporting import report

  class GreetingsOption(
    BsbOption,
    name="greeting",
    script=("greeting",),
    env=("BSB_GREETING",),
    cli=("g", "greet"),
    action=True,
  ):
    def get_default(self):
      return "Hello World! The weather today is: optimal modelling conditions."

    def action(self, namespace):
      # Actions are run before the CLI options such as verbosity take global effect.
      # Instead we can read or write the command namespace and act accordingly.
      if namespace.verbosity >= 2:
        report(self.get(), level=1)

  __plugin__ = GreetingsOption

In setup.py (assuming the above module is importable as ``my_pkg.greetings``)::

  "entry_points": {
    "bsb.options" = ["greeting = my_pkg.greetings"]
  }

After installing the setup with pip your option will be available::

  $> bsb
  $> bsb -g
  $> bsb -v 2 -g
  Hello World! The weather today is: optimal modelling conditions.
  $> export BSB_GREETING="2 PIs walk into a conference..."
  $> bsb -v 2 --greet
  2 PIs walk into a conference...
