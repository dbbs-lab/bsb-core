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

<<<<<<<<<<<<<<SOME SEGWAY TO THE CONFIGURATION HERE>>>>>>>>>>>>>>>>>>>>>

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
  for i in range(5):
    purkinje.placement.planar_density = i * 20 / 100 * max_density
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
  for i in range(5):
    purkinje.placement.planar_density = i * 20 / 100 * max_density
    scaffold.compile_network()

    # Maybe do something useful with the network here

    scaffold.reset_network_cache()

Network compilation
-------------------

With a proper configuration a new network model with cells in plausible,
non-intersecting positions, connected to each other according to specified
placement and connection algorithms can be created. This process is known as
``compilation``::

  from scaffold.scaffold import Scaffold
  from scaffold.config import JSONConfig

  config = JSONConfig(file="my_config.json")
  scaffold = Scaffold(config)
  scaffold.compile_network()

The configuration object can be modified before compilation to allow full
flexibility on top of the configuration mechanisms.

Network simulation
------------------

Simulations can be ran in a managed way using::

  scaffold.run_simulation(name)

For more control over the interface with the simulator, or finer control of the
configuration, the process can be split into parts. The adapter to the interface
of the simulator can be ejected and it's configuration can be modified::

  adapter = scaffold.create_adapter(name)
  adapter.devices["input_stimulation"].parameters["rate"] = 40

You can then use this adapter to prepare the simulator for the configured simulation::

  simulator = adapter.prepare()

After preparation the simulator is primed, but can still be modified. For
example to create 5 extra cells with the NEST simulator::

  cells = simulator.Create("iaf_cond_alpha", 5)
  print(cells)

You'll notice that the IDs of those cells won't start at 1 as would be the case
for an empty simulation, because the ``prepare`` statement has already created
cells in the simulator.

After custom interfacing with the simulator, the adapter can be used to run the simulation::

  adapter.simulate()


================
Using Cell Types
================

Cell types are obtained by name using `scaffold.get_cell_type(name)`. And the
associated cells either currently in the network cache or in persistent storage
can be fetched with `scaffold.get_cells_by_type(name)`. The columns of such
a set is are the scaffold id of the cell, followed by the type id and the xyz
position.

A collection of all cell types can be retrieved with `scaffold.get_cell_types()`::

  for cell_type in scaffold.get_cell_types():
    cells = scaffold.get_cells_by_type(cell_type.name)
    for cell in cells:
      print("Cell id {} of type {} at position {}.".format(cell[0], cell[1], cell[2:5]))
