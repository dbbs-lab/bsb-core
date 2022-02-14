############
Introduction
############

The command line interface is composed of a collection of `pluggable <Writing your own
commands>`_ commands. Open up your favorite terminal and enter the ``bsb --help`` command
to verify you correctly installed the software.

Each command can give command specific arguments, options or set `global options
<options_list>`_. For example:

.. code-block:: bash

  # Without arguments, relying on project settings defaults
  bsb compile
  # Providing the argument
  bsb compile my_config.json
  # Overriding the global verbosity option
  bsb compile --verbosity 4

=========================
Writing your own commands
=========================

You can add your own commands into the CLI by creating a class that inherits from
:class:`bsb.cli.commands.BsbCommand` and registering its module as a ``bsb.commands``
entry point. You can provide a ``name`` and ``parent`` in the class argument list.
If no parent is given the command is added under the root ``bsb`` command:

.. code-block:: python

  # BaseCommand inherits from BsbCommand too but contains the default CLI command
  # functions already implemented.
  from bsb.commands import BaseCommand

  class MyCommand(BaseCommand, name="test"):
    def handler(self, namespace):
      print("My command was run")

  class MySubcommand(BaseCommand, name="sub", parent=MyCommand):
    def handler(self, namespace):
      print("My subcommand was run")

In setup.py (assuming the above module is importable as ``my_pkg.commands``)::

  "entry_points": {
    "bsb.commands" = ["my_commands = my_pkg.commands"]
  }

After installing the setup with pip your command will be available::

  $> bsb test
  My command was run
  $> bsb test sub
  My subcommand was run
