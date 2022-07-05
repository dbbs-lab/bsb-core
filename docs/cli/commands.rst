################
List of commands
################


.. note::
  Parameters included between angle brackets are example values, parameters between square
  brackets are optional, leave off the brackets in the actual command.

Every command starts with: ``bsb [OPTIONS]``, where ``[OPTIONS]`` can
be any combination of :doc:`BSB options </cli/options>`.

.. _bsb_new:

Create a project
================

.. code-block:: bash

  bsb [OPTIONS] new <project-name> <parent-folder> [--quickstart] [--exists]

Creates a new project directory at ``folder``. You will be prompted to fill in some
project settings.

* ``project-name``: Name of the project, and of the directory that will be created for it.
* ``parent-folder``: Filesystem location where the project folder will be created.
* ``quickstart``: Generates an exemplary project with basic config that can be compiled.
* ``exists``: With this flag, it is not an error for the ``parent-folder`` to exist.

.. _bsb_make_config:

Create a configuration
======================

.. code-block:: bash

  bsb [OPTIONS] make-config <template.json> <output.json> [--path <path1> <path2 ...>]

Create a configuration in the current directory, based off the template. Specify
additional paths to search extra locations, if the configuration isn't a registered
template.

* ``template.json``: Filename of the template to look for. Templates can be registered
  through the ``bsb.config.templates`` :doc:`plugin endpoint </dev/plugins>`. Does not
  need to be a json file, just a file that can be parsed by your installed parsers.
* ``output.json``: Filename to be created.
* ``--path``: Give additional paths to be searched for the template here.

.. _bsb_compile:

Compiling a network
===================

.. code-block:: bash

  bsb [OPTIONS] compile [my-config.json] [COMPILE-FLAGS]

Compiles a network architecture according to the configuration. If no configuration is
specified, the project default is used.

* ``my-config.json``: Path to the configuration file that should be compiled. If omitted
  the :ref:`project configuration <project_settings>` path is used.

.. rubric:: Flags

* ``-x``, ``-y``, ``-z``: Size hints of the network.

* ``-o``, ``--output``: Output the result to a specific file. If omitted the value from
  the configuration, the project default, or a timestamped filename are used.

* ``-p``, ``--plot``: Plot the created network.

.. _storage_control:

.. rubric:: Storage flags

These flags decide what to do with existing data.

* ``-w``, ``--clear``: Clear all data found in the storage object, and overwrite with new
  data.

* ``-a``, ``--append``: Append the new data to the existing data.

* ``-r``, ``--redo``: Clear all data that is involved in the strategies that are being
  executed, and replace it with the new data.

.. rubric:: Phase flags

These flags control which phases and strategies to execute or ignore.

* ``--np``, ``--skip-placement``: Skip the placement phase.
* ``--nap``, ``--skip-after-placement``: Skip the after-placement phase.
* ``--nc``, ``--skip-connectivity``: Skip the connectivity phase.
* ``--nac``, ``--skip-after-connectivity``: Skip the after-connectivity phase.
* ``--skip``: Name of a strategy to skip. You may pass this flag multiple times, or give
  a comma separated list of names.

* ``--only``: Name of a strategy to run, skipping all other strategies. You may pass this
  flag multiple times, or give a comma separated list of names.

.. _bsb_simulate:

Run a simulation
================

.. code-block:: bash

  bsb [OPTIONS] simulate <path/to/netw.hdf5> <sim-name>

Run a simulation from a compiled network architecture.

* ``path/to/netw.hdf5``: Path to the network file.
* ``sim-name``: Name of the simulation.

.. _bsb_cache:

Check the global cache
======================

.. code-block:: bash

  bsb [OPTIONS] cache [--clear]

Check which files are currently cached, and optionally clear them.
