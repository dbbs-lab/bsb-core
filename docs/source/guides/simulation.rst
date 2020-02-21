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


****************
NEST simulations
****************
NEST simulations can be configured setting the attribute ``simulator`` to ``nest``.
The basic simulation properties can be set through the attributes:

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



Cell models
===========
For each cell in the ``cell_types`` dictionary, it is possible to specify:
* ``cell_model``: NEST neuron model, if not using the ``default_neuron_model``
* ``parameters``: neuron model parameters that are common to the NEST neuron models that could be used.



Connection models
=================


Plastic connections
===================
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
