###############
Getting Started
###############

This guide aims to get your first model running with the bare minimum steps. If you'd like
to familiarize yourself with the core concepts and get a more top level understanding
first, check out the :doc:`top-level-guide` before you continue.

There are 2 ways of building models using the Brain Scaffold Builder (BSB), the first is
through **configuration**, the second is **scripting**. The 2 methods complement each
other so that you can load the general model from a configuration file and then layer on
more complex steps under your full control in a Python script.

=================
Creating a config
=================

Let's create a bare configuration file called ``network_configuration.json``:

.. code-block:: json

  {
    "storage": {
      "engine": "hdf5",
      "root": "my_network.hdf5"
    },
    "network": {
      "x": 50,
      "y": 200,
      "z": 50
    },
    "regions": {

    },
    "partitions": {

    },
    "cell_types": {

    },
    "placement": {

    },
    "connectivity": {

    },
    "simulations": {

    }
  }

This configuration file declares that we'll be creating a model in an HDF5
format as a file called ``my_network.hdf5`` and that the estimated scale of the
network is a square column of 50 by 200 by 50 micrometer.

We can already use the CLI tool or a script to create an empty network from this
configuration:

.. code-block:: python

  from bsb.core import Scaffold
  from bsb.config import from_json

  config = from_json("config.json")
  scaffold = Scaffold(config)
  # Compile the empty network.
  scaffold.compile_network()
  # Your mostly empty HDF5 file `my_network.hdf5` should appear

Or to achieve the same thing from the CLI:

.. code-block:: bash

  scaffold -c=config.json compile

Defining a volume
=================

In order to generate output with actual cell positions and connection we have to
define regions, what partitions they are made of, which cell populations exist,
how to place them and how to connect them.

Let's begin by adding a ``cortex`` region, with a ``base_layer``:

.. code-block:: json

  {
    "regions": {
      "cortex": {
        "origin": [0.0, 0.0, 0.0]
      }
    },
    "partitions": {
      "base_layer": {
        "thickness": 600,
        "region": "cortex",
        "z_index": 0
      }
    }
  }

The default behavior of a region is to take on the shape of the network and to
arrange its partitions within this volume. This configuration will result in a
single 50x200x50 partition. For more information on how to create more
complicated network topologies see the :doc:`/topology/intro`.

Defining cell types and placement
=================================

Next we can start defining cell types and how to place them in said partition:

.. code-block:: json

  {
    "cell_types": {
      "cell_type_A": {
        "spatial": {
          "radius": 7,
          "count": 10
        }
      },
      "cell_type_B": {
        "spatial": {
          "radius": 2.5,
          "density": 1e-3
        }
      }
    },
    "placement": {
      "cls": "bsb.placement.ParticlePlacement",
      "cell_types": ["cell_type_A", "cell_type_B"],
      "partitions": ["base_layer"]
    }
  }

Cell types define how to represent cells in space (as points, morphologies,
ROIs, ...). The ``placement`` nodes can then use this information to place cell
type(s) into partition(s) using a ``PlacementStrategy`` class, either one
provided out of the box by the BSB or your own (see
:doc:`/guides/placement-strategies`). The :class:`.placement.ParticlePlacement`
just considers the cells as somas and bumps them around as repelling particles
until there is no overlap between the somas.

At this point we can repeat the CLI command with the plotting flag ``-p`` to
look at the result:

.. code-block:: bash

	bsb compile -c=config.json -p

Defining connection types
=========================

.. code-block:: json

  {
    "connectivity": {
      "A_to_B": {
        "cls": "bsb.connectivity.AllToAll",
        "from_type": "cell_type_A",
        "to_type": "cell_type_B"
      }
    }
  }

<More conn info>

At this point compiling the network generates cell positions and connections and
we can move to the simulation stage.

Defining simulations
====================

.. code-block:: json

  {
    "simulations": {
      "nrn_example": {
        "simulator": "neuron",
        "temperature": 32,
        "resolution": 0.1,
        "duration": 1000,
        "cell_models": {

        },
        "connection_models": {

        },
        "devices": {

        }
      },
      "nest_example": {
        "simulator": "nest",
        "default_neuron_model": "iaf_cond_alpha",
        "default_synapse_model": "static_synapse",
        "duration": 1000.0,
        "resolution": 0.1,
        "modules": ["my_extension_module"],
        "cell_models": {

        },
        "connection_models": {

        },
        "devices": {

        }
      }
    }
  }

The definition of simulations begins with chosing a simulator, either ``nest``,
``neuron`` or ``arbor``. Each simulator has their adapter and each adapter its
own requirements, see :doc:`/simulation/adapters.rst`. All of them share the
commonality that they configure ``cell_models``, ``connection_models`` and
``devices``.

Defining cell models
--------------------

A cell model is used to describe a member of a cell type during a simulation.

NEURON
~~~~~~

A cell model is described by loading external ``arborize.CellModel`` classes:

.. code-block:: json

  {
    "cell_models": {
      "cell_type_A": {
        "model": "dbbs_models.GranuleCell",
        "record_soma": true,
        "record_spikes": true
      },
      "cell_type_B": {
        "model": "dbbs_models.PurkinjeCell",
        "record_soma": true,
        "record_spikes": true
      }
    }
  }

This example dictates that during simulation setup, any member of
``cell_type_A`` should be created by importing and using
``dbbs_models.GranuleCell``. Documentation incomplete, see ``arborize`` docs ad
interim.

NEST
~~~~

In NEST the cell models need to correspond to the available models in NEST and
parameters can be given:

.. code-block:: json

  {
    "cell_models": {
      "cell_type_A": {
        "neuron_model": "iaf_cond_alpha",
        "parameters": {
          "t_ref": 1.5,
          "C_m": 7.0,
          "V_th": -41.0,
          "V_reset": -70.0,
          "E_L": -62.0,
          "I_e": 0.0,
          "tau_syn_ex": 5.8,
          "tau_syn_in": 13.61,
          "g_L": 0.29
        }
      },
      "cell_type_B": {
        "neuron_model": "iaf_cond_alpha",
        "parameters": {
          "t_ref": 1.5,
          "C_m": 7.0,
          "V_th": -41.0,
          "V_reset": -70.0,
          "E_L": -62.0,
          "I_e": 0.0,
          "tau_syn_ex": 5.8,
          "tau_syn_in": 13.61,
          "g_L": 0.29
        }
      }
    }
  }

Defining connection models
--------------------------

Connection models represent the connections between cells during a simulation.

NEURON
~~~~~~

Once more the connection models are predefined inside of ``arborize`` and they
can be referenced by name:

.. code-block:: json

  {
    "connection_models": {
      "A_to_B": {
        "synapses": ["AMPA", "NMDA"]
      }
    }
  }

NEST
~~~~

Connection models need to match the available connection models in NEST:

.. code-block:: json

  {
    "connection_models": {
      "A_to_B": {
        "synapse_model": "static_synapse",
        "connection": {
          "weight":-0.3,
          "delay": 5.0
        },
        "synapse": {
          "static_synapse": {}
        }
      }
    }
  }

Defining devices
----------------

NEURON
~~~~~~

In NEURON an assortment of devices is provided by the BSB to send input, or
record output. See :doc:`/simulation/neuron/devices.rst` for a complete list.
Some devices like voltage and spike recorders can be placed by requesting them
on cell models using :guilabel:`record_soma` or :guilabel:`record_spikes`.

In addition to voltage and spike recording we'll place a spike generator and a
voltage clamp:

.. code-block:: json

  {
    "devices": {
      "stimulus": {
        "io": "input",
        "device": "spike_generator",
        "targetting": "cell_type",
        "cell_types": ["cell_type_A"],
        "synapses": ["AMPA"],
        "start": 500,
        "number": 10,
        "interval": 10,
        "noise": true
      },
      "voltage_clamp": {
        "io": "input",
        "device": "voltage_clamp",
        "targetting": "cell_type",
        "cell_types": ["cell_type_B"],
        "cell_count": 1,
        "section_types": ["soma"],
        "section_count": 1,
        "parameters": {
          "delay": 0,
          "duration": 1000,
          "after": 0,
          "voltage": -63
        }
      }
    }
  }

The voltage clamp targets 1 random ``cell_type_B`` which is a bit awkward, but
either the ``targetting`` (docs incomplete) or the ``labelling`` system (docs
incomplete) can help you target exactly the right cells.

Running a simulation
====================

Simulations can be run through the CLI tool, or for more control through the
``bsb`` library.

CLI simulations
---------------

After network compilation you should have obtained an ``hdf5`` network file. It
contains all the required information, including a copy of its configuration. We
can use all of that to set up a "hands off" simulation, it tells the framework
to:

* Read the network file
* Read the simulation configuration
* Translate the simulation configuration to the simulator
* Create all cells, connections and devices
* Run the simulation
* Collect all the output

.. code-block:: bash

  bsb simulate my_network.hdf5 my_sim_name

Script simulations
------------------

To have more control simulations can also be set up from Python. For example,
this is what a parameter sweep would look like:

.. code-block:: python

  from bsb.core import from_hdf5
  import dbbs_models
  import nrnsub

  # Read the network file
  network = from_hdf("my_network.hdf5")

  # Make sure each NEURON simulation is ran in isolation
  @nrnsub.isolate
  def sweep(param):
    # Get an adapter to the simulation
    adapter = network.create_adapter("my_sim_name")
    # Modify the parameter to sweep
    dbbs_models.GranuleCell.synapses["AMPA"]["U"] = param
    # Prepare simulator & instantiate all the cells and connections
    simulation = adapter.prepare()
    # Optionally perform more custom operations before the simulation here.
    # Run the simulation
    adapter.simulate(simulation)
    # Optionally perform more operations or even additional simulation here.
    # Collect all results in an HDF5 file and get the path to it.
    result_file = adapter.collect_output()
    return result_file

  for i in range(11):
    # Sweep parameter from 0 to 1 in 0.1 increments
    result_file = sweep(i / 10)

.. note::

	For a more extensive introduction to the possibilities of configuring model
	components, check out the :doc:`/config/intro`!

=======================
Extending the framework
=======================

The framework allows you to plug in user code pretty much anywhere. Neat.

Here's how you do it (theoretically):

1. Identify which **interface** you need to extend. An interface is a programming concept
  that lets you take one of the objects of the framework and define some functions on it.
  The framework has predefined this set of functions and expects you to provide them.
  Interfaces in the framework are always classes.
1. Create a class that inherits from that interface and implement the required and/or
  interesting looking functions of its public API (which will be specified).
1. Refer to the class from the configuration by its importable module name, or use a
  :ref:`classmap`.

With a quick example, there's the ``MorphologySelector`` interface, which lets you specify
how a subset of the available morphologies should be selected for a certain group of
cells:

1. The interface is ``bsb.morphologies.MorphologySelector`` and the docs specify it has
  a ``validate(self, morphos)`` and ``pick(self, morpho)`` function.

.. code-block:: python

  from bsb.objects.cell_type import MorphologySelector
  from bsb import config

  @config.node
  class MySizeSelector(MorphologySelector):
    min_size = config.attr(type=float, default=20)
    max_size = config.attr(type=float, default=50)

    def validate(self, morphos):
      if not all("size" in m.get_meta() for m in morphos):
        raise Exception("Missing size metadata for the size selector")

    def pick(self, morpho):
      meta = morpho.get_meta()
      return meta["size"] > self.min_size and meta["size"] < self.max_size

Assuming that that code is in a ``select.py`` file relative to the working directory you
can now access

.. code-block:: json

  {
    "selector": "select.MySizeSelector",
    "min_size": 30,
    "max_size": 50
  }

For the model to work after you've extended the framework you have to include the Python
code, or even better, become an author of a plugin! |:heart_eyes:|

Installing plugins
==================

The fanciness doesn't end there, you can also (hopefully, somewhere in the future) install
community plugins, and they will provide extensions through Python's packaging system,
readily importable as for example ``their_plugin.selectors.TheirSizeSelector``.
