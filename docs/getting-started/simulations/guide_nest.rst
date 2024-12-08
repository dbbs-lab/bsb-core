.. _simulation-guide:

##############################
Run your first NEST simulation
##############################

.. note::

    This guide is a continuation of the
    :doc:`Getting Started guide </getting-started/getting-started_reconstruction>`.

Install requirements
====================

`NEST <https://nest-simulator.readthedocs.io/en/stable/>`_ is one of the supported
simulators of the BSB.
As for the other simulator, its adapter code is stored in a separate repository:
`bsb-neuron <https://github.com/dbbs-lab/bsb-nest>`_

So, you would need to install it with pip:

.. code-block:: bash

    pip install bsb-nest

Unfortunately, the NEST simulator at the moment can not be installed directly by pip, but
fortunately NEST provides
`tutorials <https://nest-simulator.readthedocs.io/en/stable/installation/index.html>`_
to install it in your python environment.

Make sure that you can both load BSB and NEST before continuing any further:

.. code-block:: python

    import nest
    import bsb

Configuration of the simulation
===============================

In this tutorial, we assume that you have successfully reconstructed a network with BSB.
We will now guide you through the process of configuring a simulation with BSB for your network.

We want here to put the circuit reconstructed in a steady state with a low basal activity.

Let's start by configuring the global simulation parameters.
These include the :guilabel:`simulator` to be used; in our example, we are setting it to
use NEST.
Additionally, you need to define the :guilabel:`resolution` (the time step of the simulation in milliseconds)
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

    If you are using Python code, we assume that you load your Scaffold and Configuration
    from your compiled network file:

    .. code-block:: python

        scaffold = from_storage("network.hdf5")
        config = scaffold.configuration

Cells Models
------------
For each **cell type population** of your network, you will need to assign a **point neuron model** to
determine how these cells will behave during the simulation (i.e., their inner equations).
The keys given in the :guilabel:`cell_models` should correspond to one of the :guilabel:`cell_types` of your
configuration.

.. note::

    If a certain ``cell_type`` does not have a corresponding ``cell_model`` then no cells of that type will be
    instantiated in the network.

Here, we choose one of the simplest NEST models, the
`Integrate-and-Fire neuron model <https://nest-simulator.readthedocs.io/en/v3.8/models/iaf_cond_alpha.html>`_:

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

NEST provides default parameters for each point neuron model, so we do not need to add anything.
Still, you can modify certain parameters, by setting its :guilabel:`constants` dictionary:

.. tab-set-code::

    .. code-block:: json

      "cell_models": {
        "base_type": {
          "model": "iaf_cond_alpha",
          "constants": {
            "t_ref": 1.5,
            "V_m": -62.0
          }
        },

    .. code-block:: python

        config.simulations["basal_activity"].cell_models=dict(
          base_type={"model":"iaf_cond_alpha", dict(t_ref=1.5, V_m=-62.0)},
        )


Connection Models
-----------------

For each connection type of your network, you also need to define a model describing its synapses' dynamics.
Similar to the :guilabel:`cell_models` block, for each :guilabel:`connection_model` you should use a key
that corresponds to a ``ConnectivitySet`` created during reconstruction (as explained in the previous
:doc:`section </getting-started/getting-started_reconstruction>`).
In this example, we assign the ``static_synapse`` model to the connections :guilabel:`A_to_B`.

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

For this model, the synapse model needs ``weight`` and ``delay`` parameters that are set to 100 and 1 ms,
respectively.

Devices
-------

In the :guilabel:`devices` block, include all interfaces you wish to use for interacting with the network.
These devices correspond typically to stimulators and measurement instruments.

Use the :guilabel:`device` key to select the type of device.
We also introduce here the :guilabel:`targetting` concept for the devices: This configuration node allows you to
filter elements of your neuron circuit to which you want to link your devices (see the targetting section on
:doc:`this page </simulation/intro>` for more details).

.. tab-set-code::

    .. code-block:: json

            "devices": {
                    "background_noise": {
                      "device": "poisson_generator",
                      "rate": 20,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": ["base_type"]
                      },
                      "weight": 40,
                      "delay": 1
                    },
                    "base_layer_record": {
                      "device": "spike_recorder",
                      "delay": 0.1,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": ["base_type"]
                      }
                    },
                    "top_layer_record": {
                      "device": "spike_recorder",
                      "delay": 0.1,
                      "targetting": {
                        "strategy": "cell_model",
                        "cell_models": ["top_type"]
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
                        "cell_models": ["base_type"]
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

In our example, we add a ``poisson_generator`` that simulates cells spiking at ``20`` Hz.
These latter "cells" are each connected one ``top_type`` cell and transmit their spike events with a delay
of `1` ms and the weight of the connection is ``40``.
We also introduce a ``spike_recorder`` to store the spike events of the cell populations.

Final configuration file
------------------------

.. tab-set-code::

  .. literalinclude:: ../configs/guide-nest.yaml
    :language: yaml

  .. literalinclude:: ../configs/guide-nest.json
    :language: json

  .. literalinclude:: /../examples/tutorials/nest-simulation.py
    :language: python
    :lines: 1-45


Running the Simulation
======================

Simulations are separated from the reconstruction pipeline (see the
:doc:`top level guide </getting-started/top-level-guide>`),
which means you do not need to recompile your network to add a simulation to your stored Configuration.
In this example, we only modified the ``Configuration`` in the :guilabel:`simulations` block but this updates were
not been saved in the network file.
So, you need to update your file, using either the ``reconfigure`` command or the ``store_active_config`` method.

.. tab-set-code::

  .. code-block:: bash

    bsb reconfigure network.hdf5 network_configuration.json

  .. code-block:: python

    storage = scaffold.storage
    storage.store_active_config(config)

Once this is done, create a folder in which to store your simulation results:

.. code-block:: bash

    mkdir simulation-results

You can now run your simulation:

.. tab-set-code::

  .. code-block:: bash

    bsb simulate network.hdf5 basal_activity -o simulation-results

  .. code-block:: python

        from bsb import from_storage

        scaffold = from_storage("network.hdf5")
        result = scaffold.run_simulation("basal_activity")
        result.write("simulation-results/basal_activity.nio", "ow")

The results of the simulation will be stored in the ``"simulation-results"`` folder.

.. note::
    If you run the simulation with the command line interface, the name of the output nio file is randomized by BSB.

For more detailed information about simulation modules,
please refer to the :doc:`simulation section </simulation/intro>`.

Congratulations, you simulated your first BSB reconstructed network with NEST!

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1


    .. grid-item-card:: :octicon:`fold-up;1em;sd-text-warning` Analyze your Results
        :link: guide_analyze_results
        :link-type: ref

        How to extract your data.

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Make custom components
       :link: guide_components
       :link-type: ref

       Learn how to write your own components to e.g. place or connect cells.

    .. grid-item-card:: :octicon:`gear;1em;sd-text-warning` Learn about components
       :link: main-components
       :link-type: ref

       Explore more about the main components.

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
        :link: examples
        :link-type: ref

        Explore more advanced examples
