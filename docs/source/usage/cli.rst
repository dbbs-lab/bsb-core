######################
Command Line Interface
######################


*****************************
List of command line commands
*****************************

.. note::
  Parameters included between angle brackets are example values, parameters between square
  brackets are optional, leave off the brackets in the actual command.

Every command starts with: ``bsb [OPTIONS]``, where ``[OPTIONS]`` can
be any combination of :doc:`BSB options </usage/options>`.

``new <project-name> <parent-folder>``
======================================

Creates a new project directory at ``folder``. You will be prompted to fill in some
project settings.

* ``project-name``: Name of the project, and of the directory that will be created for it.
* ``parent-folder``: Filesystem location where the project folder will be created.

``make-config <template.json> <output.json> [--path <path1> <path2 ...>]``
==========================================================================

Create a configuration in the current directory, based off the template. Specify
additional paths to search extra locations, if the configuration isn't a registered
template.

* ``template.json``: Filename of the template to look for. Templates can be registered
  through the ``bsb.config.templates`` :doc:`plugin endpoint </dev/plugins>``. Does not
  need to be a json file, just a file that can be parsed by your installed parsers.
* ``output.json``: Filename to be created.
* ``--path``: Give additional paths to be searched for the template here.

``compile [my-config.json] [-p] [-o file]``
====================================

Compiles a network architecture according to the configuration. If no configuration is
specified, the project default is used.

* ``-p``: Plot the created network.
* ``-o``, ``--output``: Output the result to a specific file. If omitted the value from
  the configuration, the project default, or a timestamped filename are used.

``simulate <netw.hdf5> <sim-name>``
===================================

Run a simulation from a compiled network architecture.

* ``name``: Name of the simulation.
* ``--hdf5``: Path to the compiled network architecture.
* ``-rc``, ``--reconfigure``: The path to a new configuration file for the HDF5
  file.
