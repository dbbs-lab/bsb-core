.. _simulations:

###################
Simulating networks
###################

.. The BSB manages simulations by deferring as soon as possible to the simulation backends.
.. Each simulator has good reasons to make their design choices, befitting of their
.. simulation paradigm. These choices lead to divergence in how simulations are described,
.. and each simulator has their own niche functions. This means that if you are already
.. familiar with a simulator, writing simulation config should feel familiar, on top of that
.. the BSB is able to offer you access to each simulator's full set of features. The downside
.. is that you're required to write a separate simulation config block per backend.
..
.. Now, let's get started.

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


.. rubric:: Parallel simulations

To parallelize any BSB task prepend the MPI command in front of the BSB CLI command, or
the Python script command:

.. code-block:: bash

  mpirun -n 4 bsb simulate my_network.hdf5 your_simulation
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

* ``default_neuron_model``: default model used for all ``cell_models``, unless indicated
  otherwise in the ``neuron_model`` attribute of a specific cell model.
* ``default_synapse_model``: default model used for all ``connection_models`` (e.g.
  ``static_synapse``), unless differently indicated in the ``synapse_model`` attribute of
  a specific connection model.
* ``modules``: list of NEST extension modules to be installed.

.. code-block:: json

  {
    "simulations": {
      "first_simulation": {
        "simulator": "nest",
        "default_neuron_model": "iaf_cond_alpha",
        "default_synapse_model": "static_synapse",
        "duration": 1000,
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
In the ``cell_models`` attribute, it is possible to specify simulation-specific properties
for each cell type:

* ``cell_model``: NEST neuron model, if not using the ``default_neuron_model``. See the
  available models in the
  `NEST documentation <https://nest-simulator.readthedocs.io/en/latest/models/neurons.html>`_
* ``parameters``: neuron model parameters that are common to the NEST neuron models that
  could be used, including:

  * ``t_ref``: refractory period duration [ms]
  * ``C_m``: membrane capacitance [pF]
  * ``V_th``: threshold potential [mV]
  * ``V_reset``: reset potential [mV]
  * ``E_L``: leakage potential [mV]

Then, neuron model specific parameters can be indicated in the attributes corresponding to
the model names:

* ``iaf_cond_alpha``:

  * ``I_e``: endogenous current [pA]
  * ``tau_syn_ex``: time constant of excitatory synaptic inputs [ms]
  * ``tau_syn_in``: time constant of inhibitory synaptic inputs [ms]
  * ``g_L``: leaky conductance [nS]

* ``eglif_cond_alpha_multisyn``:

  * ``Vmin``: minimum membrane potential [mV]
  * ``Vinit``: initial membrane potential [mV]
  * ``lambda_0``: escape rate parameter
  * ``tau_V``: escape rate parameter
  * ``tau_m``: membrane time constant [ms]
  * ``I_e``: endogenous current [pA]
  * ``kadap``: adaptive current coupling constant
  * ``k1``: spike-triggered current decay
  * ``k2``: adaptive current decay
  * ``A1``: spike-triggered current update [pA]
  * ``A2``: adaptive current update [pA]
  * ``tau_syn1``, ``tau_syn2``, ``tau_syn3``: time constants of synaptic inputs at the 3
    receptors [ms]
  * ``E_rev1``, ``E_rev2``, ``E_rev3``: reversal potential for the 3 synaptic receptors
    (usually set to 0mV for excitatory and -80mV for inhibitory synapses) [mV]
  * ``receptors``: dictionary specifying the receptor number for each input cell to the
    current neuron

Example
~~~~~~~
Configuration example for a cerebellar Golgi cell. In the ``eglif_cond_alpha_multisyn``
neuron model, the 3 receptors are associated to synapses from glomeruli, Golgi cells and
Granule cells, respectively.

.. code-block:: json

  {
    "cell_models": {
      "golgi_cell": {
        "parameters": {
          "t_ref": 2.0,
          "C_m": 145.0,
          "V_th": -55.0,
          "V_reset": -75.0,
          "E_L": -62.0
        },
        "iaf_cond_alpha": {
          "I_e": 36.75,
          "tau_syn_ex": 0.23,
          "tau_syn_in": 10.0,
          "g_L": 3.3
        },
        "eglif_cond_alpha_multisyn": {
          "Vmin": -150.0,
          "Vinit": -62.0,
          "lambda_0": 1.0,
          "tau_V":0.4,
          "tau_m": 44.0,
          "I_e": 16.214,
          "kadap": 0.217,
          "k1": 0.031,
          "k2": 0.023,
          "A1": 259.988,
          "A2":178.01,
          "tau_syn1":0.23,
          "tau_syn2": 10.0,
          "tau_syn3": 0.5,
          "E_rev1": 0.0,
          "E_rev2": -80.0,
          "E_rev3": 0.0,
          "receptors": {
            "glomerulus": 1,
            "golgi_cell": 2,
            "granule_cell": 3
          }
        }
      }
    }
  }

Connection models
-----------------


Plasticity
----------
The default synapse model for connection models is usually set to ``static_synapse``.

For plastic synapses, it is possible to choose between:

#. Homosynaptic plasticity models (e.g. ``stdp_synapse``) where weight changes depend on
    pre- and postsynaptic spike times


#. Heterosynaptic plasticity models (e.g. ``stdp_synapse_sinexp``), where spikes of an
    external teaching population trigger the weight change. In this case, a device called
    "volume transmitter" is created for each postsynaptic neuron, collecting the spikes
    from the teaching neurons.


For a full set of available synapse models, see `the NEST documentation
<https://nest-simulator.readthedocs.io/en/latest/models/synapses.html>`_

For the plastic connections, specify the attributes as follows:

* ``plastic``: set to ``true``.
* ``hetero``: set to ``true`` if using an heterosynaptic plasticity model.
* ``teaching``: Connection model name of the teaching connection for heterosynaptic
  plasticity models.
* ``synapse_model``: the name of the NEST synapse model to be used. By default, it is the
  model specified in the ``default_synapse_model`` attribute of the current simulation.
* ``synapse``: specify the parameters for each one of the synapse models that could be
  used for that connection.

.. note::
  If the ``synapse_model`` attribute is not specified, the ``default_synapse_model`` will
  be used (``static``). Using synapse models without plasticity - such as ``static`` -
  while setting the ``plastic`` attribute to ``true`` will lead to errors.

Example
~~~~~~~

.. code-block:: json

  {
    "connection_models": {
      "parallel_fiber_to_purkinje": {
        "plastic": true,
        "hetero": true,
        "teaching": "io_to_purkinje",
        "synapse_model": "stdp_synapse_sinexp",
        "connection": {
          "weight": 0.007,
          "delay": 5.0
        },
        "synapse": {
          "static_synapse": {},
          "stdp_synapse_sinexp": {
            "A_minus": 0.5,
            "A_plus": 0.05,
            "Wmin": 0.0,
            "Wmax": 100.0
          }
        }
      },

      "purkinje_to_dcn": {
        "plastic": true,
        "synapse_model": "stdp_synapse",
        "connection": {
          "weight":-0.4,
          "delay": 4.0
        },
        "synapse": {
          "static_synapse": {},
          "stdp_synapse": {
            "tau_plus":30.0,
            "alpha": 0.5,
            "lambda": 0.1,
            "mu_plus": 0.0,
            "mu_minus": 0.0,
            "Wmax": 100.0
          }
        }
      }
    }
  }

Devices
-------

======
NEURON
======


Cell models
-----------

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
