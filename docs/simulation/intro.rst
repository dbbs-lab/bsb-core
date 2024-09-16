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

All of the above is simulation backend specific and is covered in the specific sections:

 * :doc:`NEST </simulation/nest>`.
 * :doc:`NEURON </simulation/neuron>`.
 * :doc:`ARBOR </simulation/arbor>`.

Running Simulations
-------------------

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

