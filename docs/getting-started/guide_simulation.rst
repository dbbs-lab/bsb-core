   .. _simulation-guide:

########################
Do your first simulation
########################

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
of the simplest NEST models, the `exponential integrate-and-fire neuron model <https://nest-simulator.readthedocs.io/en/v3.8/models/aeif_cond_exp.html>`_:

.. tab-set-code::

    .. code-block:: json

         "cell_models": {
            "base_type": {
              "model": "aeif_cond_exp"
            },
            "top_type": {
              "model": "aeif_cond_exp"
            }
          },

    .. code-block:: python

        config.simulations["basal_activity"].cell_models=dict(
          base_type={"model":"aeif_cond_exp"},
          top_type={"model":"aeif_cond_exp"}
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
              "weight": 1,
              "delay": 1
            }
        }
      },

    .. code-block:: python

        config.simulations["basal_activity"].connection_models=dict(
          A_to_B=dict(
            synapse=dict(
              model="static_synapse",
              weight=1,
              delay=1
            )
          )
        )

In this case the synapse model needs ``weight`` and ``delay`` parameters that are set to 1.

Devices
-------

In the devices block, include all interfaces you wish to use for interacting with the network,
referencing devices typically used in experiments, such as stimulators and measurement instruments.

.. tab-set-code::

    .. code-block:: json

            "devices": {
                    "background_noise": {
                      "device": "poisson_generator",
                      "rate": 5,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": [
                          "top_type"]
                      },
                      "weight": 1,
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
                    }
            }

    .. code-block:: python

            config.simulations["basal_activity"].devices=dict(
              general_noise=dict(
                      device= "poisson_generator",
                      rate= 5,
                      targetting= {
                        "strategy": "cell_model",
                        "cell_models": ["top_type"]
                      },
                      weight= 1,
                      delay= 1
              ),
              base_layer_record=dict(
                      device= "spike_recorder",
                      delay= 0.1,
                      targetting= {
                        "strategy": "cell_model",
                        "cell_models": ["base_type"]
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
        bsb simulate my_network.hdf5 basal_activity

Alternatively, if you prefer to manage the simulations using Python code:

.. code-block:: python

        from bsb import Scaffold

        my_network = Scaffold(config)
        my_network.compile()
        my_network.run_simulation("basal_activity")


For more detailed information about simulation modules,
please refer to the :doc:`simulation section </simulation/intro>`.