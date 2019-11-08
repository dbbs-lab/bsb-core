###############
Getting Started
###############

===========
First steps
===========

The scaffold provides a simple command line interface (CLI) to compile network
architectures and run simulations.

Let's try out the most basic command, using the default configuration::

  scaffold -v=3 compile

This should produce prints and generate a timestamped HDF5 file in your current
directory.

You can explore the structure of the generated output by analysing it with the
scaffold shell. Open the scaffold shell like this::

  scaffold

You can now open and view the output HDF5 file like this::

  open hdf5 <name>.hdf5
  view

.. note::
  By default the output file should be named ``scaffold_network`` followed by
  a timestamp.

This will print out the datasets and attributes in the output file. Most notably
this should give you access to the cell positions and connections.

See :doc:`/usage/cli` for a full guide.

============
First script
============

Although the scaffold package features a CLI that can perform most tasks, its
primary use case is to be included in scripts that can further customize
the scaffold with things impossible to achieve using the configuration files.

Let's go over an example first script that creates 100 networks with different
densities of Purkinje cells.

To use the scaffold in your script you should import the :class:`scaffold.scaffold.Scaffold`
and construct a new instance by passing it a :class:`scaffold.config.ScaffoldConfig`.
The only provided configuration is the :class:`scaffold.config.JSONConfig`.
To load a configuration file, construct a JSONConfig object providing the `file`
keyword argument with a path to the configuration file::

  from scaffold.scaffold import Scaffold
  from scaffold.config import JSONConfig

  config = JSONConfig(file="my_config.json")
  scaffold = Scaffold(config)

Let's find the purkinje cell configuration::

  purkinje = scaffold.get_cell_type("purkinje_cell")

The next step is to adapt the Purkinje cell density each iteration. The location
of the attributes on the Python objects mostly corresponds to their location in
the configuration file. This means
``"purkinje_cell": { "placement": { "planar_density"} }`` can be found under
``placement.planar_density``::

  max_density = purkinje.placement.planar_density
  for i in range(100):
    purkinje.placement.planar_density = i / 100 * max_density
    scaffold.compile_network()

    # Maybe do something useful with the network here

    scaffold.reset_network_cache()

.. warning::
  If you don't use ``reset_network_cache()`` between ``compile_network()`` calls
  the new cells will just be appended to the previous ones. This will lead to
  confusing result and long computation times.

Full code example
-----------------

::

  from scaffold.scaffold import Scaffold
  from scaffold.config import JSONConfig

  config = JSONConfig(file="my_config.json")
  scaffold = Scaffold(config)
  purkinje = scaffold.get_cell_type("purkinje_cell")
  max_density = purkinje.placement.planar_density
  for i in range(100):
    purkinje.placement.planar_density = i / 100 * max_density
    scaffold.compile_network()

    # Maybe do something useful with the network here

    scaffold.reset_network_cache()
