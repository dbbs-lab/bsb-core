###################
Simulating Networks
###################

The BSB offers adapters that enable you to simulate
your network using widely-used neural simulation software. Consequently, once the model is created,
it can be simulated across different software platforms without requiring modifications or adjustments.
Currently, adapters are available for NEST, NEURON, and ARBOR,
although support for ARBOR is not yet fully developed.

All simulation details are specified within the simulation block, which includes:
 * a ``simulator`` : the software chosen for the simulations.
 * set of ``cell models`` : the simulator specific representations of the network's :doc:`CellTypes </cells/intro>`
 * set of ``connection models`` :  that instruct the simulator on how to handle the :doc:`ConnectivityStrategies </connectivity/defining>` of the network
 * set of ``devices`` : define the experimental setup (such as input stimuli and recorders).

All of the above is simulation backend specific and is covered in the corresponding sections:

 * :doc:`NEST </simulation/nest>`.
 * :doc:`NEURON </simulation/neuron>`.
 * :doc:`ARBOR </simulation/arbor>`.

Running Simulations
===================

Simulations can be run through the CLI or through the ``bsb`` library for more
control:

.. tab-set-code::

    .. code-block:: bash

      bsb simulate my_network.hdf5 my_sim_name

    .. code-block:: python

        from bsb import from_storage
        network = from_storage("my_network.hdf5")
        network.run_simulation("my_sim")

When using the CLI, the framework sets up a "hands off" simulation workflow:

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

Targetting
==========

To customize our experimental setup, devices can be arranged to target specific cell populations.
In the BSB, several methods are available to filter the populations of interest.
These methods can be based on various criteria, including cell characteristics,
labels, and geometric constraints within the network volume.

The target population can be defined when a device block is created in the configuration:

.. tab-set-code::

    .. code-block:: json

        "my_new_device": {
          "device": "device_type",
          "targetting": {
            "strategy": "my_target_strategy",
          }
        }
    .. code-block:: python

        config.simulations["my_simulation_name"].devices=dict(
          my_new_device={
            "device": "device_type",
            "targetting": {
              "strategy": "my_target_strategy",
            }
          }
        )

Strategies based on cell
------------------------

``strategy name``: :guilabel:`all` . This is a basic strategy that targets all the cells in our network

Target by cell model
^^^^^^^^^^^^^^^^^^^^

``strategy name``: :guilabel:`cell_model` . This strategy targets only the cells of the specified models.
Users must provide a list of cell models to target using the attribute :guilabel:`cell_models` .

Target by id
^^^^^^^^^^^^

``strategy name``: :guilabel:`by_id` . Each cell model is assigned a numerical identifier
that can be used to select the target cells.
It is necessary to provide a list of integers representing the cell IDs with the attribute :guilabel:`ids` .


Geometric strategies
--------------------

Instead of targeting cells based on characteristics or labels,
it is possible to target a defined region using geometric constraints.

Target a Cylinder
^^^^^^^^^^^^^^^^^

``strategy name``: :guilabel:`cylinder`. This strategy targets all the cells contained within a cylinder along the defined axis.
The user must provide three attributes:

* ``origin``: A *list* of coordinates representing the base of the cylinder for each non-main axis.
* ``axis``: A character is used to specify the main axis of the cylinder. Accepted values are "x," "y," and "z," with the default set to "y."
* ``radius``: A *float* representing the radius of the cylinder.

Target a Sphere
^^^^^^^^^^^^^^^

``strategy name``: :guilabel:`sphere`. This strategy targets all the cells contained within a sphere.
The user must provide two attributes:

* ``origin``: A *list* of *float* that defines the center of the sphere.
* ``radius``: A *float* representing the radius of the sphere.

Simulation results
==================

The results of a simulation are stored in ``.nio`` files.
These files are read and written using the :doc:`Neo Python package <neo:index>`, which is specifically
designed for managing electrophysiology data. The files utilize the HDF5 (Hierarchical Data Format version 5)
structure, which organizes data in a hierarchical manner, similar to a dictionary.
The data is organized into blocks, where each block represents a recording session (i.e., a simulation block).
Each block is further subdivided into segments, with each segment representing a specific timeframe within the session.
To retrieve the blocks from a ``.nio`` file:

.. code-block:: python

  from neo import io

  neo_obj = io.NixIO("NAME_OF_YOUR_NEO_FILE.nio", mode="ro")
  blocks = neo_obj.read_all_blocks()

  for block in blocks:
    list_of_segments = blocks.segments


For more information, please refer to the :doc:`Neo documentation <neo:read_and_analyze>`.

Spike Trains
------------

Within a segment, you can access all the :class:`SpikeTrain <neo.core.SpikeTrain>` objects recorded during
that particular timeframe. A ``SpikeTrain`` represents the collection of spikes emitted by the target population,
and it includes metadata about the device name, the size of the cell population, and the IDs of the spiking cells.
This information is stored in the :guilabel:`annotations` attribute:

.. code-block:: python

  for spiketrain in segment.spiketrains:
      spiketrain_array = spiketrain.magnitude
      unit_of_measure = spiketrain.units
      device_name = spiketrain.annotations["device"]
      list_of_spiking_cell_ids = spiketrain.annotations["senders"]
      end_time_of_the_simulation = spiketrain.annotations["t_stop"]
      population_size = spiketrain.annotations["pop_size"]

.. note::

  To retrieve the spike train for a specific cell, the spike time at index i in the ``spiketrain_array[i]``
  corresponds to the cell with ID ``list_of_spiking_cell_ids[i]``.

Analog Signals
--------------

Each segment also contains an :guilabel:`analogsignals` attribute, which holds a list of Neo :class:`AnalogSignal <neo.core.AnalogSignal>` objects.
These objects contain the trace of the recorded property, along with the associated time points.
They are also annotated with information such as the device name, the type of cell recorded, and the cell ID,
which can be accessed through the :guilabel:`annotations` attribute:

.. code-block:: python

  for signal in segment.analogsignals:
    trace_of_the_signal = signal.magnitude
    unit_of_measure = signal.units
    time_signature = signal.times
    time_unit = signal.times.units
    device_name = signal.annotations['name']
    cell_type = signal.annotations['cell_type']
    cell_id = signal.annotations['cell_id']

.. note::

  Unlike the spike train case, the :guilabel:`analogsignals` attribute contains a separate ``AnalogSignal``
  object for each target of the device.

