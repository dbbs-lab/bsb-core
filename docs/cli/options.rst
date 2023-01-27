#######
Options
#######

The BSB has several global options, which can be set through a 12-factor style cascade.
The cascade goes as follows, in descending priority: script, CLI, project, env. The first
to provide a value will be used. For example, if both a CLI and env value are provided,
the CLI value will override the env value.

The script values can be set from the ``bsb.options`` module, CLI values can be passed to
the command line, project settings can be stored in ``pyproject.toml``, and env values can
be set through use of environment variables.

Using script values
-------------------

Read option values; if no script value is set, the other values are checked in cascade
order:

.. code-block:: python

  import bsb.options

  print(bsb.options.verbosity)

Set a script value; it has highest priority for the remainder of the Python process:

.. code-block:: python

  import bsb.options

  bsb.options.verbosity = 4

Once the Python process ends, the values are lost. If you instead would like to set a
script value but also keep it permanently as a project value, use store_.

Using CLI values
----------------

The second priority are the values passed through the CLI, options may appear anywhere in
the command.

Compile with verbosity 4 enabled:

.. code-block:: shell

  bsb -v 4 compile
  bsb compile -v 4

Using project values
--------------------

Project values are stored in the Python project configuration file ``pyproject.toml`` in
the ``tools.bsb`` section. You can modify the `TOML <https://toml.io/en/>`_ content in the
file, or use :func:`.options.store`:

.. _store:

.. code-block:: python

  import bsb.options

  bsb.options.store("verbosity", 4)

The value will be written to ``pyproject.toml`` and saved permanently at project level. To
read any ``pyproject.toml`` values you can use :func:`.options.read`:

.. code-block:: python

  import bsb.options

  link = bsb.options.read("networks.config_link")

Using env values
----------------

Environment variables are specified on the host machine, for Linux you can set one with
the following command:

.. code-block:: shell

  export BSB_VERBOSITY=4

This value will remain active until you close your shell session. To keep the value around
you can store it in a configuration file like ``~/.bashrc`` or ``~/.profile``.

.. _options_list:

List of options
---------------

* ``verbosity``: Determines how much output is produced when running the BSB.

  * *script*: ``verbosity``

  * *cli*: ``v``, ``verbosity``

  * *project*: ``verbosity``

  * *env*: ``BSB_VERBOSITY``

* ``force``: Enables sudo mode. Will execute destructive actions without confirmation,
  error or user interaction. Use with caution.

  * *script*: ``sudo``

  * *cli*: ``f``, ``force``

  * *project*: None.

  * *env*: ``BSB_FOOTGUN_MODE``

* ``version``: Tells you the BSB version. **readonly**

  * *script*: ``version``

  * *cli*: ``version``

  * *project*: None.

  * *env*: None.

* ``config``: The default config file to use, if omitted in commands.

  * *script*: None (when scripting, you should create a :class:`~.config.Configuration`)
    object.

  * *cli*: ``config``, usually positional. e.g. ``bsb compile conf.json``

  * *project*: ``config``

  * *env*: ``BSB_CONFIG_FILE``

.. _project_settings:

``pyproject.toml`` structure
----------------------------

The BSB's project-wide settings are all stored in ``pyproject.toml`` under ``tools.bsb``:

.. code-block:: toml

  [tools.bsb]
  config = "network_configuration.json"

========================
Writing your own options
========================

You can create your own options as a :doc:`plugin </dev/plugins>` by defining a class that
inherits from :class:`~.option.BsbOption`:

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

  # Make `GreetingsOption` available as the default plugin object of this module.
  __plugin__ = GreetingsOption

Plugins are installed by ``pip`` which takes its information from
``setup.py``/``setup.cfg``, where you can specify an entry point::

  "entry_points": {
    "bsb.options" = ["greeting = my_pkg.greetings"]
  }

After installing the setup with ``pip`` your option will be available::

  $> pip install -e .
  $> bsb
  $> bsb --greet
  $> bsb -v 2 --greet
  Hello World! The weather today is: optimal modelling conditions.
  $> export BSB_GREETING="2 PIs walk into a conference..."
  $> bsb -v 2 --greet
  2 PIs walk into a conference...

For more information on setting up plugins (even just locally) see :doc:`/dev/plugins`.
