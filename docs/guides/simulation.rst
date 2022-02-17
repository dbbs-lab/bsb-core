###########
Simulations
###########

After building the scaffold models, simulations can be run using `NEST <https://www.nest-simulator.org/>`_ or NEURON.

Simulations can be configured in the ``simulations`` dictionary of the root node of the
configuration file, specifying each simulation with its name, e.g. "first_simulation", "second_simulation":

.. code-block:: json

  {
    "simulations": {
      "first_simulation": {

      },
      "second_simulation": {

      }
    }
  }


****
NEST
****
NEST is mainly used for simulations of Spiking Neural Networks, with point neuron models.

*************
Configuration
*************
NEST simulations in the scaffold can be configured setting the attribute ``simulator`` to ``nest``.
The basic NEST simulation properties can be set through the attributes:

* ``default_neuron_model``: default model used for all ``cell_models``, unless differently indicated in the ``neuron_model`` attribute of a specific cell model.
* ``default_synapse_model``: default model used for all ``connection_models`` (e.g. ``static_synapse``), unless differently indicated in the ``synapse_model`` attribute of a specific connection model.
* ``duration``: simulation duration in [ms].
* ``modules``: list of NEST extension modules to be installed.

Then, the dictionaries ``cell_models``, ``connection_models``, ``devices``, ``entities`` specify the properties of each element of the simulation.

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

        },
        "entities": {

        }

      },
      "second_simulation": {

      }
    }
  }



Cells
=====
In the ``cell_models`` attribute, it is possible to specify simulation-specific properties for each cell type:

* ``cell_model``: NEST neuron model, if not using the ``default_neuron_model``. Currently supported models are ``iaf_cond_alpha`` and ``eglif_cond_alpha_multisyn``. Other available models can be found in the `NEST documentation <https://nest-simulator.readthedocs.io/en/latest/models/neurons.html>`_
* ``parameters``: neuron model parameters that are common to the NEST neuron models that could be used, including:

  * ``t_ref``: refractory period duration [ms]
  * ``C_m``: membrane capacitance [pF]
  * ``V_th``: threshold potential [mV]
  * ``V_reset``: reset potential [mV]
  * ``E_L``: leakage potential [mV]

Then, neuron model specific parameters can be indicated in the attributes corresponding to the model names:

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
  * ``tau_syn1``, ``tau_syn2``, ``tau_syn3``: time constants of synaptic inputs at the 3 receptors [ms]
  * ``E_rev1``, ``E_rev2``, ``E_rev3``: reversal potential for the 3 synaptic receptors (usually set to 0mV for excitatory and -80mV for inhibitory synapses) [mV]
  * ``receptors``: dictionary specifying the receptor number for each input cell to the current neuron

Example
=======
Configuration example for a cerebellar Golgi cell. In the ``eglif_cond_alpha_multisyn`` neuron model, the 3 receptors are associated to synapses from glomeruli, Golgi cells and Granule cells, respectively.

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



Connections
=========================


Simulations with plasticity
===========================
The default synapse model for connection models is usually set to ``static_synapse``.

For plastic synapses, it is possible to choose between:

1. homosynaptic plasticity models (e.g. ``stdp_synapse``) where weight changes depend on pre- and postsynaptic spike times

2. heterosynaptic plasticity models (e.g. ``stdp_synapse_sinexp``), where spikes of an external teaching population trigger the weight change. In this case, a device called "volume transmitter" is created for each postsynaptic neuron, collecting the spikes from the teaching neurons.

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
=======





Entities
========
