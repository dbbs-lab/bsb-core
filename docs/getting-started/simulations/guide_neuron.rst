################################
Run your first NEURON simulation
################################

In this example we are going to present how to configure a simulation of a multi-compartment neurons network.
To proceed, please install the following additional packages:

.. code-block:: bash

    pip install bsb-neuron[parallel]
    pip install dbbs-catalogue

For this example, we will build a network consisting of a layer of mouse
stellate cells connected through axon-dendrite overlap, using the strategy :doc:`VoxelIntersection </connectivity/connection-strategies>`.
The morphology of a custom stellate cell is provided :download:`here </getting-started/data/StellateCell.swc>`.

The network configuration should be as follows:

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 1-71

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 1-54

Now we have to configure the simulation block.
Let's start by configuring the global simulation parameters: first of all
define a :guilabel:`simulator`, then you need to define the :guilabel:`resolution`
(the time step of the simulation in milliseconds),
the :guilabel:`duration` (the total length of the simulation in milliseconds) and
the :guilabel:`temperature` (celsius unit).

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 72-76

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 56-61

Cell Models
-----------

For each **cell type** population in your network, you must assign a **NEURON model** to define the cell's behavior. This model
encapsulates all the specifications for ion channels and synapses covering all components of the cell..
Within a model the synapse parameters are defined in the :guilabel:`synapse_types` attribute, while the
parameters for ion channel mechanisms are defined in :guilabel:`cable_types`. A detailed discussion of model
characteristics is beyond the scope of this guide; therefore, a ready-to-use Stellate model is provided
:download:`here </getting-started/data/Stellate.py>`.

Save the Stellate.py file in your project folder and review its contents.
Inside, you will find a model definition called
:guilabel:`definitionStellate`, which includes all the customized parameters.
This is the object you will reference in your configuration.

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 79-83

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 62-64

Connection Models
-----------------

For each connection type of your network, you also need to define a model describing its synapses' dynamics.
Similar to the :guilabel:`cell_models` block, each :guilabel:`connection_model` you define should use a key
that corresponds to a ``ConnectivitySet`` created during reconstruction (as explained in the previous
:doc:`section </getting-started/getting-started_reconstruction>`).
In this example we have only the :guilabel:`stellate_to_stellate` connection, where we assign the synapses
defined in the model file, namely :guilabel:`AMPA`, :guilabel:`GABA`, and :guilabel:`NMDA`.

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 84-91

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 65-73

At all the synapses is assigned a :guilabel:`weight` of 0.001 and a :guilabel:`delay` (ms) of 1.

Devices
-------

In the :guilabel:`devices` block, include all interfaces you wish to use for interacting with the network.
These devices correspond typically to stimulators and measurement instruments.

In this example, a :guilabel:`spike_generator` is used to stimulate the cell with ID 0,
starting at 9 ms, with 1 spike. The stimulus targets the dendrites through AMPA and NMDA synapses.
The membrane potential is recorded using a :guilabel:`voltage_recorder`, which collects the
signal from within a 600 µm radius sphere. Synapse activity is monitored with a :guilabel:`synapse_recorder`
for the :guilabel:`AMPA` and :guilabel:`NMDA` synapses on the cell's dendrites,
within the same spherical region.

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 92-136

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 74-107

Final configuration file
------------------------

.. tab-set-code::


  .. literalinclude:: /getting-started/configs/guide-neuron.json
    :language: json

  .. literalinclude:: /../examples/tutorials/neuron-simulation.py
    :language: python

Running the Simulation
----------------------

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

    bsb simulate my_network.hdf5 basal_activity -o simulation-results

  .. code-block:: python

        from bsb import from_storage

        scaffold = from_storage("my_network.hdf5")
        result = scaffold.run_simulation("basal_activity")
        result.write("simulation-results/basal_activity.nio", "ow")

The results of the simulation will be stored in the ``"simulation-results"`` folder.

.. note::
    If you run the simulation with the command line interface, the name of the output nio file is randomized by BSB.

For more detailed information about simulation modules,
please refer to the :doc:`simulation section </simulation/intro>`.



