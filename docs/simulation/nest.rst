###################
Simulate using NEST
###################



NEST is a simulation tool used in computational neuroscience for modeling and studying the behavior of
large networks of point neurons.

To install the simulator, please refer to the installation `guide <https://nest-simulator.readthedocs.io/en/v3.8/installation/index.html>`_.

The simulation block contains all the necessary details to run the simulation.
In the configuration, you will need to specify a name and duration (in milliseconds).
For example, the following creates a simulation named ``my_simulation_name`` with a duration of 1 second:

.. tab-set-code::

    .. code-block:: json

        "simulations": {
            "my_simulation_name": {
              "simulator": "nest",
              "duration": 1000,
        }

    .. code-block:: python


        config.simulations.add("my_simulation_name",
          simulator="nest",
          duration=1000,
        )


Cell Models
===========

In the cell_models block, you specify the NEST representation for each cell type.
Each key in the block can have the following attributes:

    * :guilabel:`model`: NEST neuron model, See the available models in the NEST documentation

    * :guilabel:`constants`: parameters that are defined in the NEST neuron model. Most models include the following parameters:

        * t_ref: refractory period duration [ms]
        * C_m: membrane capacitance [pF]
        * V_th: threshold potential [mV]
        * V_reset: reset potential [mV]
        * E_L: leakage potential [mV]

For example if you have a network with two cell types:

.. tab-set-code::

    .. code-block:: json

         "cell_models": {
            "cell_type_one": {
              "model": "first_nest_model",
              "constants": { "t_ref": 2, "C_m": 15}
            },
            "cell_type_two": {
              "model": "another_nest_name",
              "constants": {"t_ref": 1}
            }
          },

    .. code-block:: python

        config.simulations["my_simulation_name"].cell_models=dict(
          cell_type_one={"model":"first_nest_model",constants={ "t_ref": 2, "C_m": 15}},
          cell_type_two={"model":"another_nest_name",constants={ "t_ref": 1}}
        )

For the first cell type, we assign the corresponding NEST neuron model named ``first_nest_model``,
with a refractory period of 2 ms and a membrane capacitance of 15 pF.
For cell_type_two, we use the NEST model named ``another_nest_name``, setting a different refractory period.

To select the most suitable model and its parameters, the NEST website provides a useful model `list <https://nest-simulator.readthedocs.io/en/v3.8/neurons/index.html>`_.

Connection Models
=================

In this block, you need to define a NEST connection module for each ConnectivitySet in your network.
The `bsb-nest <https://github.com/dbbs-lab/bsb-nest>`_ interface provides a ``NestConnection`` module that handles the set by passing arrays of
pre-synaptic and post-synaptic cells, using a ``one_to_one`` connection.
You will need to specify the synapse configuration using the key :guilabel:`synapse`, providing all required properties as a dictionary.

The available keys in the synapse specification dictionary include:

    * :guilabel:`model` : Specifies NEST ``synapse_model``.
    * :guilabel:`weight` : *float* that defines a weight for the synapse (**required**).
    * :guilabel:`delay` : *float* that defines a delay (**required**).
    * :guilabel:`receptor_type` : *int* that identifies NEST receptor types. For more details see the `receptors section <https://nest-simulator.readthedocs.io/en/v3.8/synapses/synapse_specification.html#receptor-types>`_ .
    * :guilabel:`constants` : Any parameters specific to the selected synapse model.

.. tab-set-code::

    .. code-block:: json

         "connection_models": {
            "connect_A_to_B": {
              "synapse" : {
                "model": "static_synapse",
                "weight": 0.5,
                "delay": 1
              }
            },
            "connect_B_to_A": {
              "synapse": {
                "model": "bernoulli_synapse",
                "weight": 1,
                "delay": 1,
                "constants":{"p_transmit":0.5}
              }
            }
          },

    .. code-block:: python

        config.simulations["my_simulation_name"].connection_models=dict(
          connect_A_to_B=dict(synapse=dict(
              model="static_synapse",
              weight=0.5,
              delay=1
              )
          )
          connect_B_to_A=dict(synapse=dict(
              model="bernoulli_synapse",
              weight=1,
              delay=1,
              constants={"p_transmit":0.5}
              )
          )
        )

In this example, for the ConnectivitySet named ``connect_A_to_B``, we assign a ``static_synapse``,
while for the second set, a ``bernoulli_synapse`` is chosen.
All available built-in synapse models are listed in the `NEST guide <https://nest-simulator.readthedocs.io/en/v3.8/synapses/index.html>`_.

Devices
=======

In this block, you can specify the devices for your experimental setup.
NEST provides two types of devices: *recording* devices, for extracting information from a simulation,
and *stimulation* devices, for delivering stimuli.

The ``bsb-nest`` module provides interfaces for NEST devices through the ``NestDevice`` object.
To properly configure a device, you need to specify three attributes:

   * :guilabel:`weight` : *float* specifying the connection weight between the device and its target (required).
   * :guilabel:`delay` : *float* specifying the transmission delay between the device and its target (required).
   * :guilabel:`targeting` : Specifies the targets of the device, which can be a population or a NEST rule.

For example, to create a device named ``my_new_device`` of class ``device_type``, with a weight of 1
and a delay of 0.1 ms, targeting the population of ``my_cell_model``:

.. tab-set-code::

    .. code-block:: json

        "my_new_device": {
          "device": "device_type",
          "weight": 1,
          "delay": 0.1,
          "targetting": {
            "strategy": "cell_model",
            "cell_models": ["my_cell_model"]
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_new_device={
            "device": "device_type",
            "weight": 1,
            "delay": 0.1,
            "targetting": {
              "strategy": "cell_model",
              "cell_models": ["my_cell_model"]
            }
          }
        )

Stimulation devices
-------------------

These type of devices are used to inject signals into a network. Currently two types of
stimuli are supported by ``bsb-nest``. NEST guide provides more details about `stimulation devices <https://nest-simulator.readthedocs.io/en/v3.8/devices/stimulate_the_network.html#stimulate-network>`_.

Create a DC Generator
^^^^^^^^^^^^^^^^^^^^^

This generator provides a constant DC input to the connected node. It can be configured by passing three attributes:

    * :guilabel:`amplitude` : *float* that is the amplitude of the current in mV (**required**).
    * :guilabel:`start` : *float* define a starting time in ms.
    * :guilabel:`stop` : *float* define an arresting time in ms.

.. tab-set-code::

    .. code-block:: json

        "my_new_device": {
          "device": "dc_generator",
          "amplitude": 20,
          "weight": 1,
          "delay": 0.1,
          "targetting": {
            "strategy": "cell_model",
            "cell_models": ["my_cell_model"]
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_new_device={
            "device": "dc_generator",
            "amplitude": 20,
            "weight": 1,
            "delay": 0.1,
            "targetting": {
              "strategy": "cell_model",
              "cell_models": ["my_cell_model"]
            }
          }
        )

Create a Poisson Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^

This generator simulates a neuron firing with Poisson statistics, generating a unique spike train
for each of its targets. You need to specify the frequency of the generator,
in terms of the mean firing rate (Hz), using the :guilabel:`rate` key.

.. tab-set-code::

    .. code-block:: json

        "my_new_device": {
          "device": "poisson_generator",
          "rate": 10,
          "weight": 1,
          "delay": 0.1,
          "targetting": {
            "strategy": "cell_model",
            "cell_models": ["my_cell_model"]
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_new_device={
            "device": "poisson_generator",
            "rate": 10,
            "weight": 1,
            "delay": 0.1,
            "targetting": {
              "strategy": "cell_model",
              "cell_models": ["my_cell_model"]
            }
          }
        )

Recording Devices
-----------------

These are the devices which are used to observe the state of network nodes. Currently ``bsb-nest``
support two types of recorders.

Add a Spike Recorder
^^^^^^^^^^^^^^^^^^^^

Is one of the most basic collector device, which records all spikes it receives from neurons that are connected to it.
An example of usage with a delay of 0.1 could be:

.. tab-set-code::

    .. code-block:: json

        "my_collector": {
          "device": "spike_recorder",
          "delay": 0.1,
          "targetting": {
            "strategy": "cell_model",
            "cell_models": ["my_cell_model"]
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_collector={
            "device": "spike_recorder",
            "delay": 0.1,
            "targetting": {
              "strategy": "cell_model",
              "cell_models": ["my_cell_model"]
            }
          }
        )

.. note::
 For this device, the :guilabel:`weight` attribute it is set to 1.

Add a Multimeter
^^^^^^^^^^^^^^^^

This type of device allows you to record analog values from neurons.
Unlike the spike recorder, which collects neuron outputs, this device inspects its
targets at defined time intervals to sample the quantities of interest.
To properly add a multimeter for your target neurons you have to specify:

    * :guilabel:`properties` : List of properties to record in the Nest model.
    * :guilabel:`units` : List of properties' units.

Potential recordable properties are given in the corresponding section of the NEST model documentation.

.. tab-set-code::

    .. code-block:: json

        "my_sampler": {
          "device": "multimeter",
          "delay": 0.1,
          "properties": ["V_m"],
          "units": ["mV"],
          "targetting": {
            "strategy": "cell_model",
            "cell_models": ["my_cell_model"]
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_sampler={
            "device": "multimeter",
            "delay": 0.1,
            "properties": ["V_m"],
            "units": ["mV"],
            "targetting": {
              "strategy": "cell_model",
              "cell_models": ["my_cell_model"]
            }
          }
        )

In this example we add a multimeter to sample the membrane voltage of ``my_cell_model``.

.. note::
 For this device, the :guilabel:`weight` attribute it is set to 1.