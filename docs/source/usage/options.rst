#######
Options
#######

The BSB has several global options, which can be set through a 12-factor style cascade.
The cascade goes as follows, in descending priority: script, CLI, project, env. The first
to provide a value will be used. For example, if both a CLI and env value are provided,
the CLI value will override the env value.

The script values can be set from the ``bsb.options`` module, CLI values can be passed to
the command line, project settings can be stored in ``pyproject.toml``, and env values can
be set as environment variables.

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

``pyproject.toml`` structure
----------------------------

The BSB's project-wide settings are all stored in ``pyproject.toml`` under ``tools.bsb``:

.. code-block:: toml

  [tools.bsb]
  config = "network_configuration.json"

  [tools.bsb.networks]
  config_link = ["sys", "network_configuration.json", "always"]
  morpho_link = ["sys", "morphologies.h5", "changes"]
