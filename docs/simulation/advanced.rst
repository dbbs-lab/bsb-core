#####################
Simulation Components
#####################

The `Simulation` object encapsulates all the parameters necessary to adapt a reconstructed network to a specific simulator.
It primarily handles the conversion of cell types and connectivity into simulator-compatible formats and prepares the experimental configuration.

A `Simulation` is defined by the following attributes:

* ``simulator``:*str* - Specifies the simulation software to be used.
* ``duration``: *float* - Total duration of the simulation in milliseconds.
* ``cell_models``: *dict* - Contains simulator-specific representations of the network's  :doc:`CellTypes </cells/intro>`.
* ``connection_models``: *dict* - Provides instructions for handling the network’s :doc:`ConnectivityStrategies </connectivity/defining>`.
* ``devices``: *dict* - Lists the simulation devices to be included.
* ``post_prepare``: *Callable* - A hook that is executed after the simulation has been prepared.




Simulator Adapters
==================

The `SimulatorAdapter` is the core abstraction responsible for adapting BSB simulation data and
execution flow to the specifics of a target simulator.

This class manages the simulation pipeline, which consists of the following stages:

**Prepare** → **Post Prepare** → **Run** → **Collect**

As an abstract base class, it is designed to be extended to implement simulator-specific behavior.
At a minimum, custom adapters must implement the :guilabel:`prepare()` and :guilabel:`run()` methods:

* :guilabel:`prepare()`: Initializes the simulation by configuring parameters, creating cell_models and connection_models, and invoking the :guilabel:`implement()` method on the simulation devices.
* :guilabel:`run()`: Executes the simulation. Typically, this involves stepping through simulation time intervals until the defined duration is reached, using the simulator’s solver to compute the system’s evolution.

The **Post Prepare** and **Collect** phases do not need to be implemented. After the :guilabel:`prepare()` method completes, the adapter will execute any functions specified in the ``post_prepare`` hook.
Once the simulation ends, the **Collect** phase gathers and finalizes results.

Adapter Iterators
-----------------
To monitor simulation progress at defined intervals, you can use the `AdapterProgress` class.
This utility handles iteration over simulation time steps.
Example usage:

.. code-block:: python

            def run(self, *simulations: "Simulation"):

                duration = max(sim.duration for sim in simulations)
                progress = AdapterProgress(duration)
                my_interval=1
                for oi, i in progress.steps(step=my_interval):
                    my_solver(oi,i) ## call the solver from time oi to time i
                    tick = progress.tick(i)
                progress.complete()

* :guilabel:`steps(step=...)`: Yields time intervals of the specified step size (default is 1 ms).
* :guilabel:`tick(i)`: Returns a ``SimpleNamespace`` object with current progress information.

If intermediate result collection is needed before simulation ends, use the `AdapterCheckpoint` class.
It coordinates checkpoints from all registered devices and merges them into a unified schedule.

.. code-block:: python

            def run(self, *simulations: "Simulation"):

                sim = simulations["sim_name"]
                duration = sim.duration
                progress = AdapterProgress(duration)
                my_interval=1
                checkpoint = AdapterCheckpoint(simulations)
                optimal_interval = checkpoint.suitable_step(my_interval)
                for oi, i in progress.steps(step=optimal_interval):
                    my_solver(oi,i) ## call the solver from time oi to time i
                    tick = progress.tick(i)

                    if checkpoint.get_status(i):
                        self.simdata[sim].result.flush()

                progress.complete()

* :guilabel:`suitable_step(interval)`: Suggests an optimal interval compatible with the defined checkpoints.
* :guilabel:`get_status(time)`: Returns True if a checkpoint has been reached at the given simulation time.