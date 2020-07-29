###############
Getting Started
###############

===========
First steps
===========

The scaffold provides a simple command line interface (CLI) to compile network
architectures and run simulations.

Let's try out the most basic command, using the default configuration::

  scaffold -v=3 compile -x=200 -z=200

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

The scaffold exposes many general circuit builder features through a JSON
configuration interface. By adapting values in the configuration a wide range
of networks can be obtained. Extending the cerebellum model with new cell types
can be achieved simply by adding new cell type and connection configuration
objects to the configuration file. By building new configuration files the
placement and connection strategies used to construct the cerebellum scaffold
model could be leveraged to build any general brain area topology.

You can use the default configuration of the mouse cerebellum as a starting
point for your own scaffold model::

  scaffold make-config my_config.json

You can modify values in there and create a network from it like so::

  scaffold -c=my_config compile -p

Open the configuration file in your favorite editor and reduce the simulation
volume::

  "network_architecture": {
    "simulation_volume_x": 400.0, # For local single core 150 by 150 is doable.
    "simulation_volume_z": 400.0,

See :doc:`/configuration` for more on the configuration interface. Complex
brain scaffolds can be constructed purely using these files, but there might be
cases where it isn't enough, that's why it's also possible to augment the
configuration with Python scripting:

============
First script
============

Although the scaffold package features a CLI that can perform most tasks, its
primary use case is to be included in scripts that can further customize
the scaffold with things impossible to achieve using the configuration files.

Let's go over an example first script that creates 5 networks with different
densities of Purkinje cells.

To use the scaffold in your script you should import the :class:`bsb.core.Scaffold`
and construct a new instance by passing it a :class:`bsb.config.ScaffoldConfig`.
The only provided configuration is the :class:`bsb.config.JSONConfig`.
To load a configuration file, construct a JSONConfig object providing the `file`
keyword argument with a path to the configuration file::

  from bsb.core import Scaffold
  from bsb.config import JSONConfig
  from bsb.reporting import set_verbosity

  config = JSONConfig(file="my_config.json")
  set_verbosity(3) # This way we can follow what's going on.
  scaffold = Scaffold(config)

.. note::
  The verbosity is 1 by default, which only displays errors. You could also add
  a `verbosity` attribute to the root node of the `my_config.json` file to set
  the verbosity.

Let's find the purkinje cell configuration::

  purkinje = scaffold.get_cell_type("purkinje_cell")

The next step is to adapt the Purkinje cell density each iteration. The location
of the attributes on the Python objects mostly corresponds to their location in
the configuration file. This means that::

  "purkinje_cell": {
    "placement": {
      "planar_density": 0.045,
      ...
    },
    ...
  }

will be stored in the Python ``CellType`` object under
``purkinje.placement.planar_density``::

  max_density = purkinje.placement.planar_density
  for i in range(5):
    purkinje.placement.planar_density = i * 20 / 100 * max_density
    scaffold.compile_network()

    scaffold.plot_network_cache()

    scaffold.reset_network_cache()

.. warning::
  If you don't use ``reset_network_cache()`` between ``compile_network()`` calls
  the new cells will just be appended to the previous ones. This might lead to
  confusing results.

Full code example
-----------------

::

  from bsb.core import Scaffold
  from bsb.config import JSONConfig
  from bsb.reporting import set_verbosity

  config = JSONConfig(file="my_config.json")
  set_verbosity(3) # This way we can follow what's going on.
  scaffold = Scaffold(config)

  purkinje = scaffold.get_cell_type("purkinje_cell")
  max_density = purkinje.placement.planar_density

  for i in range(5):
    purkinje.placement.planar_density = i * 20 / 100 * max_density
    scaffold.compile_network()

    scaffold.plot_network_cache()

    scaffold.reset_network_cache()

Network compilation
-------------------

``compilation`` is the process of creating an output containing the constructed
network with cells placed according to the specified placement strategies and
connected to each other according to the specified connection strategies::

  from bsb.core import Scaffold
  from bsb.config import JSONConfig

  config = JSONConfig(file="my_config.json")

  # The configuration provided in the file can be overwritten here.
  # For example:
  config.cell_types["some_cell"].placement.some_parameter = 50
  config.cell_types["some_cell"].plotting.color = ENV_PLOTTING_COLOR

  scaffold = Scaffold(config)
  scaffold.compile_network()

The configuration object can be freely modified before compilation, although
values that depend on eachother - e.g. layers in a stack - will not update each
other.

Network simulation
------------------

Simulations can be executed from configuration in a managed way using::

  scaffold.run_simulation(name)

This will load the simulation configuration associated with ``name`` and create
an adapter for the simulator. An adapter translates the scaffold configuration
into commands for the simulator. In this way scaffold adapters are able to
prepare simulations in external simulators such as NEST or NEURON for you. After
the simulator is prepared the simulation is ran.

For more control over the interface with the simulator, or finer control of
the configuration, the process can be split into parts. The adapter to the
interface of the simulator can be ejected and its configuration can be
modified::

  adapter = scaffold.create_adapter(name)
  adapter.devices["input_stimulation"].parameters["rate"] = 40

You can then use this adapter to prepare the simulator for the configured
simulation::

  simulator = adapter.prepare()

After preparation the simulator is primed, but can still be modified directly
accessing the interface of the simulator itself. For example to create 5 extra
cells in a NEST simulation::

  cells = simulator.Create("iaf_cond_alpha", 5)
  print(cells)

You'll notice that the IDs of those cells won't start at 1 as would be the case
for an empty simulation, because the ``prepare`` statement has already created
cells in the simulator.

After custom interfacing with the simulator, the adapter can be used to run the
simulation::

  adapter.simulate()


================
Using Cell Types
================

Cell types are obtained by name using `bsb.get_cell_type(name)`. And the
associated cells either currently in the network cache or in persistent storage
can be fetched with `bsb.get_cells_by_type(name)`. The columns of such
a set are the scaffold id of the cell, followed by the type id and the xyz
position.

A collection of all cell types can be retrieved with `bsb.get_cell_types()`::

  for cell_type in scaffold.get_cell_types():
    cells = scaffold.get_cells_by_type(cell_type.name)
    for cell in cells:
      print("Cell id {} of type {} at position {}.".format(cell[0], cell[1], cell[2:5]))
