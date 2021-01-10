###############
Getting Started
###############

===========
First steps
===========

The scaffold provides a simple command line interface (CLI) to compile network
architectures and run simulations.

To start, let's create ourselves a project directory and a template configuration::

  mkdir my_brain
  cd my_brain
  bsb make-config

This should produce a template configuration with a single layer and cell type in it. You
can generate a network from it using the compile command::

  bsb -c network_configuration.json compile -p

.. note::

	You can leave off the ``-c`` (or ``--config``) flag in this case as
	``network_configuration.json`` is the default config that ``bsb compile`` will look for.
	The ``-p`` (or ``--plot``) flag will plot your network afterwards

You can explore the structure of the generated output by analysing it with the
bsb shell or an HDF5 viewer like HDFCompass or HDFViewer. Open the bsb shell like this::

  bsb

You can now open and view the output HDF5 file like this::

  open hdf5 <name>.hdf5
  view

.. note::
  By default the output file will be named ``scaffold_network`` followed by
  a timestamp.

This will print out the datasets and attributes in the output file. Most notably
this should give you access to the cell positions and connections.

See :doc:`/usage/cli` for a full guide.

The scaffold exposes many general circuit builder features through a JSON configuration
interface. By adapting values in the configuration a wide range of networks can be
obtained. Extending the template coonfiguration with new cell types can be achieved by
adding new cell type and connection configuration objects to the configuration file.

See :doc:`/configuration` for more on the configuration interface. Complex
brain scaffolds can be constructed purely using these files, but there might be
cases where it isn't enough, that's why it's also possible to augment the
configuration with Python scripting.

============
First script
============

Although the scaffold package features a CLI that can perform most tasks, the package's
primary use case is to be included in scripts that can further customize
the scaffold with things impossible to achieve using the configuration files.

Let's go over an example first script that creates 5 networks with different
densities of ``base_type``.

To use the scaffold in your script you should import the :class:`bsb.core.Scaffold`
and construct a new instance by passing it a :class:`bsb.config.ScaffoldConfig`.
The only provided configuration is the :class:`bsb.config.JSONConfig`.
To load a configuration file, construct a JSONConfig object providing the `file`
keyword argument with a path to the configuration file::

  from bsb.core import Scaffold
  from bsb.config import JSONConfig
  from bsb.reporting import set_verbosity

  config = JSONConfig(file="network_configuration.json")
  set_verbosity(3) # This way we can follow what's going on.
  scaffold = Scaffold(config)

.. note::
  The verbosity is 1 by default, which only displays errors. You could also add
  a `verbosity` attribute to the root node of the `my_config.json` file to set
  the verbosity.

Let's find the ``base_type`` cell configuration::

  base_type = scaffold.get_cell_type("base_type")

The next step is to adapt the Purkinje cell density each iteration. The location
of the attributes on the Python objects mostly corresponds to their location in
the configuration file. This means that::

  "base_type": {
    "placement": {
      "density": 3.9e-4,
      ...
    },
    ...
  }

will be stored in the Python ``CellType`` object under
``base_type.placement.density``::

  max_density = base_type.placement.density
  for i in range(5):
    base_type.placement.density = i * 20 / 100 * max_density
    scaffold.compile_network()

    scaffold.plot_network_cache()

    scaffold.reset_network_cache()

.. warning::
  If you don't use ``reset_network_cache()`` between ``compile_network()`` calls,
  the new cells will just be appended to the previous ones. This might lead to
  confusing results.

Full code example
-----------------

::

  from bsb.core import Scaffold
  from bsb.config import JSONConfig
  from bsb.reporting import set_verbosity

  config = JSONConfig(file="network_configuration.json")
  set_verbosity(3) # This way we can follow what's going on.
  scaffold = Scaffold(config)

  base_type = scaffold.get_cell_type("base_type_cell")
  max_density = base_type.placement.density

  for i in range(5):
    base_type.placement.density = i * 20 / 100 * max_density
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

  config = JSONConfig(file="network_configuration.json")

  # The configuration provided in the file can be overwritten here.
  # For example:
  config.cell_types["some_cell"].placement.some_parameter = 50
  config.cell_types["some_cell"].plotting.color = ENV_PLOTTING_COLOR

  scaffold = Scaffold(config)
  scaffold.compile_network()

The configuration object can be freely modified before compilation, although
values that depend on eachother - i.e. layers in a stack - will not update each
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
