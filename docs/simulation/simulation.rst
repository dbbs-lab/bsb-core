.. _simulations:

###################
Simulating networks
###################

Simulations can be run through the CLI:

.. code-block:: bash

  bsb simulate my_network.hdf5 my_sim_name

or through the ``bsb`` library for more
control. When using the CLI, the framework sets up a "hands off" simulation workflow:

* Read the network file
* Read the simulation configuration
* Translate the simulation configuration to the simulator
* Create all cells, connections and devices
* Run the simulation
* Collect all the output

When you use the library, you can set up more complex workflows, such as parameter sweeps:

.. literalinclude:: ../../examples/simulation/parameter_sweep.py
  :language: python

.. rubric:: Parallel simulations

To parallelize any BSB task prepend the MPI command in front of the BSB CLI command, or
the Python script command:

.. code-block:: bash

  mpirun -n 4 bsb simulate my_network.hdf5 my_sim_name
  mpirun -n 4 python my_simulation_script.py

Where ``n`` is the number of parallel nodes you'd like to use.

=============
Configuration
=============

Each simulation config block needs to specify which :guilabel:`simulator` they use. Valid
values are ``arbor``, ``nest`` or ``neuron``. Also included in the top level block are the
:guilabel:`duration`, :guilabel:`resolution` and :guilabel:`temperature` attributes:

.. code-block:: json

  {
    "simulations": {
      "my_arbor_sim": {
        "simulator": "arbor",
        "duration": 2000,
        "resolution": 0.025,
        "temperature": 32,
        "cell_models": {

        },
        "connection_models": {

        },
        "devices": {

        }
      }
    }
  }

The :guilabel:`cell_models` are the simulator specific representations of the network's
:class:`cell types <.cell_types.CellType>`, the :guilabel:`connection_models` of the
network's :class:`connectivity types <.connectivity.strategy.ConnectionStrategy>` and the
:guilabel:`devices` define the experimental setup (such as input stimuli and recorders).
All of the above is simulation backend specific and is covered per simulator below.

=====
Arbor
=====

Cell models
-----------

The keys given in the :guilabel:`cell_models` should correspond to a ``cell type`` in the
network. If a certain ``cell type`` does not have a corresponding ``cell model`` then no
cells of that type will be instantiated in the network. Cell models in Arbor should refer
to importable ``arborize`` cell models. The Arborize model's ``.cable_cell`` factory will
be called to produce cell instances of the model:

.. code-block:: json

  {
    "cell_models": {
      "cell_type_A": {
        "model": "my.models.ModelA"
      },
      "afferent_to_A": {
        "relay": true
      }
    }
  }

.. note::

  *Relays* will be represented as ``spike_source_cells`` which can, through the connectome
  relay signals of other relays or devices. ``spike_source_cells`` cannot be the target of
  connections in Arbor, and the framework targets the targets of a relay instead, until
  only ``cable_cells`` are targeted.

Connection models
-----------------

todo: doc

.. code-block:: json

  {
    "connection_models": {
      "aff_to_A": {
        "weight": 0.1,
        "delay": 0.1
      }
    }
  }

Devices
-------

``spike_generator`` and ``probes``:

.. code-block:: json

  {
    "devices": {
      "input_stimulus": {
        "device": "spike_generator",
        "explicit_schedule": {
          "times": [1,2,3]
        },
        "targetting": "cell_type",
        "cell_types": ["mossy_fibers"]
      },
      "all_cell_recorder": {
        "targetting": "representatives",
        "device": "probe",
        "probe_type": "membrane_voltage",
        "where": "(uniform (all) 0 9 0)"
      }
    }
  }

todo: doc & link to targetting

====
NEST
====

.. rubric:: Additional root attributes:

* ``modules``: list of NEST extension modules to be installed.

.. code-block:: json

  {
    "simulations": {
      "first_simulation": {
        "simulator": "nest",
        "duration": 1000,
        "resolution": 1.0,
        "modules": ["cerebmodule"],

        "cell_models": {

        },
        "connection_models": {

        },
        "devices": {

        }
      },
      "second_simulation": {

      }
    }
  }

Cell models
-----------
In the ``cell_models`` block, you specify the simulator representation
for each cell type. Each key in the block can have the following attributes:

* ``model``: NEST neuron model, See the
  available models in the
  `NEST documentation <https://nest-simulator.readthedocs.io/en/stable/models/index.html>`_
* ``constants``: neuron model parameters that are common to the NEST neuron models that
  could be used, including:

  * ``t_ref``: refractory period duration [ms]
  * ``C_m``: membrane capacitance [pF]
  * ``V_th``: threshold potential [mV]
  * ``V_reset``: reset potential [mV]
  * ``E_L``: leakage potential [mV]

Example
~~~~~~~
Configuration example for a cerebellar Golgi cell. In the ``eglif_cond_alpha_multisyn``
neuron model, the 3 receptors are associated to synapses from glomeruli, Golgi cells and
Granule cells, respectively.

.. code-block:: json

  {
    "cell_models": {
      "golgi_cell": {
        "constants": {
          "t_ref": 2.0,
          "C_m": 145.0,
          "V_th": -55.0,
          "V_reset": -75.0,
          "E_L": -62.0
        }
      }
    }
  }

Connection models
-----------------

Devices
-------

======
NEURON
======


Cell models
-----------

By default the NEURON adapter uses an ``ArborizedCellModel``, which loads
external ``arborize`` definition to instantiate cells.

.. code-block:: json

  {
    "cell_models": {
      "cell_type_A": {
        "model": "dbbs_models.GranuleCell"
      },
      "cell_type_B": {
        "model": "dbbs_models.PurkinjeCell"
      }
    }
  }

This example dictates that during simulation setup, any member of
``cell_type_A`` should be created by importing and using
``dbbs_models.GranuleCell``. Documentation incomplete, see ``arborize`` docs ad
interim.

Connection models
-----------------

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

Devices
-------

Devices send input, or record output.

Here we'll place a spike generator and a voltage clamp:

.. code-block:: json

  {
    "devices": {
      "stimulus": {
        "device": "spike_generator",
        "targetting": {
            "strategy": "cell_model",
            "cell_models": ["cell_type_A"]
        },
        "synapses": ["AMPA"],
        "start": 500,
        "number": 10,
        "interval": 10,
        "noise": true
      },
      "voltage_clamp": {
        "device": "voltage_clamp",
        "targetting": {
            "strategy": "cell_model",
            "cell_models": ["cell_type_A"]
            "count": 1,
        },
        "location": {
            "strategy": "soma"
        },
        "parameters": {
          "delay": 0,
          "duration": 1000,
          "after": 0,
          "voltage": -63
        }
      }
    }
  }
