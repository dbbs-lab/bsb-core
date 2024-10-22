   .. _simulation-guide:

#########################
Run your first simulation
#########################

This section assumes you are already familiar with network construction. It will guide you through configuring
a simulation for your network. If you need assistance with network setup,
please refer to the :doc:`getting started guide </getting-started/getting-started_reconstruction>`.

Once the network is built, the next step is to configure the simulation parameters.
Start by specifying the :guilabel:`simulator` to be used. Here we presents the `NEST <https://nest-simulator.readthedocs.io/en/stable/installation/index.html>`_
simulator but other simulators can also be selected. Additionally, you need to define the :guilabel:`resolution` (the time step of the simulation in milliseconds)
and the :guilabel:`duration` (the total length of the simulation in milliseconds).
Therefore, your simulation block should be structured as follows:

.. tab-set-code::

    .. code-block:: json

        "simulations": {
            "basal_activity": {
              "simulator": "nest",
              "resolution": 0.1,
              "duration": 5000,
              "cell_models": {
              },
              "connection_models": {
              },
              "devices":{
              }
        }

    .. code-block:: python

        config.simulations.add("basal_activity",
          simulator="nest",
          resolution=0.1,
          duration=5000,
          cell_models={},
          connection_models={},
          devices={}
        )

.. note::

    If you are using Python code, we assume that all network blocks are already
    configured within a ``Configuration`` object named  :guilabel:`config`.


Cells Models
------------
The simulator needs a model to determine how cells will behave during the simulation.
The keys given in the :guilabel:`cell_models` should correspond to a ``cell type`` in the
network. If a certain ``cell type`` does not have a corresponding ``cell model`` then no
cells of that type will be instantiated in the network. For our case we choose one
of the simplest NEST models, the `simple leaky integrate-and-fire neuron model <https://nest-simulator.readthedocs.io/en/v3.8/models/iaf_cond_alpha.html>`_:

.. tab-set-code::

    .. code-block:: json

         "cell_models": {
            "base_type": {
              "model": "iaf_cond_alpha"
            },
            "top_type": {
              "model": "iaf_cond_alpha"
            }
          },

    .. code-block:: python

        config.simulations["basal_activity"].cell_models=dict(
          base_type={"model":"iaf_cond_alpha"},
          top_type={"model":"iaf_cond_alpha"}
        )

Connection Models
-----------------

The simulator also requires information about the types of connections to use.
Similar to the cell model block, each connection model you define should use a key that corresponds to a ``connectivity set`` present in the network.
In this example, we add a ``static_synapse`` connection to the connectivity :guilabel:`A_to_B`.

.. tab-set-code::

    .. code-block:: json

      "connection_models": {
        "A_to_B": {
            "synapse": {
              "model": "static_synapse",
              "weight": 100,
              "delay": 1
            }
        }
      },

    .. code-block:: python

        config.simulations["basal_activity"].connection_models=dict(
          A_to_B=dict(
            synapse=dict(
              model="static_synapse",
              weight=100,
              delay=1
            )
          )
        )

In this case the synapse model needs ``weight`` and ``delay`` parameters that are set to 100 and 1, rispectively.

Devices
-------

In the devices block, include all interfaces you wish to use for interacting with the network,
referencing devices typically used in experiments, such as stimulators and measurement instruments.

.. tab-set-code::

    .. code-block:: json

            "devices": {
                    "background_noise": {
                      "device": "poisson_generator",
                      "rate": 20,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": [
                          "top_type"]
                      },
                      "weight": 40,
                      "delay": 1
                    },
                    "base_layer_record": {
                      "device": "spike_recorder",
                      "delay": 0.1,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": [
                          "base_type"
                        ]
                      }
                    },
                    "top_layer_record": {
                      "device": "spike_recorder",
                      "delay": 0.1,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": [
                          "top_type"
                        ]
                      }
                    }
            }

    .. code-block:: python

            config.simulations["basal_activity"].devices=dict(
              general_noise=dict(
                      device= "poisson_generator",
                      rate= 20,
                      targetting= {
                        "strategy": "cell_model",
                        "cell_models": ["top_type"]
                      },
                      weight= 40,
                      delay= 1
              ),
              base_layer_record=dict(
                      device= "spike_recorder",
                      delay= 0.1,
                      targetting= {
                        "strategy": "cell_model",
                        "cell_models": ["base_type"]
                      }
              ),
              top_layer_record=dict(
                      device= "spike_recorder",
                      delay= 0.1,
                      targetting= {
                        "strategy": "cell_model",
                        "cell_models": ["top_type"]
                      }
              )
            )


Using the :guilabel:`device` key, you select the type of device to use, and with :guilabel:`targetting`,
you specify the target objects of the device.
In our example, we add a ``poisson_generator`` to stimulate the top layer cells and use a ``spike_recorder`` to record the activity of the base layer cells.

Running the Simulation
----------------------

Once the configuration file is complete it should be compiled producing a HDF5 network file,
this file will be used to run simulations through the CLI:

.. code-block:: bash

        bsb compile -v 3 my_configuration.json
        bsb simulate my_network.hdf5 basal_activity -f simulation-results

Alternatively, if you prefer to manage the simulations using Python code:

.. code-block:: python

        from bsb import Scaffold

        scaffold = Scaffold(config)
        scaffold.compile(true)
        result = scaffold.run_simulation("basal_activity")
        result.write("simulation-results.nio", "ow")


For more detailed information about simulation modules,
please refer to the :doc:`simulation section </simulation/intro>`.

Recap
-----

.. tab-set-code::

  .. literalinclude:: configs/guide-simulation.yaml
    :language: yaml

  .. literalinclude:: configs/guide-simulation.json
    :language: json

  .. literalinclude:: configs/guide-simulation.py
    :language: python

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1


    .. grid-item-card:: :octicon:`fold-up;1em;sd-text-warning` Analyze Results
	    :link: guide_analyze_results
	    :link-type: ref

	    How to extract your data.

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Custom components
       :link: components
       :link-type: ref

       Learn how to write your own components to e.g. place or connect cells.
