.. _simulations:

Simulations
===========

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
        "modules": ["my_extension_module"],
        "cell_models": {

        }
      }
    }
  }

The definition of simulations begins with chosing a simulator, either ``nest``,
``neuron`` or ``arbor``. Each simulator has their adapter and each adapter its
own requirements, see :doc:`/simulation/adapters`. All of them share the
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
record output. See :doc:`/simulation/neuron/devices` for a complete list.
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
--------------------

Simulations can be run through the CLI tool, or for more control through the ``bsb``
library. When using the CLI, the framework sets up a "hands off" simulation:

* Read the network file
* Read the simulation configuration
* Translate the simulation configuration to the simulator
* Create all cells, connections and devices
* Run the simulation
* Collect all the output

.. code-block:: bash

  bsb simulate my_network.hdf5 my_sim_name

When you use the library, you can set up more complex workflows, for example, this is a
parameter sweep that loops and modifies the release probability of the AMPA synapse in the
cerebellar granule cell:

.. literalinclude:: ../../examples/simulation/parameter_sweep.py
  :language: python


Parallel simulations
--------------------

To parallelize any task the BSB can execute you can prepend the MPI command in front of
the BSB CLI command, or the Python script command:

.. code-block:: bash

  mpirun -n 4 bsb simulate my_network.hdf5 your_simulation
  mpirun -n 4 python my_simulation_script.py

Where ``n`` is the number of parallel nodes you'd like to use.
