#####################
Simulate using NEURON
#####################

NEURON is a simulator tool used in the computational neuroscience community to model and study the dynamics
of multicomportamental neurons.

This simulator must be installed as a dependency when installing the `bsb-neuron <https://github.com/dbbs-lab/bsb-neuron>`_
package. If you are installing NEURON separately, please refer to the official installation
`documentation <https://nrn.readthedocs.io/en/8.2.6/install/install.html>`_ for guidance.

The simulation block contains all the necessary components to run the simulation.
In the configuration, you will need to specify a name and duration (in milliseconds).
For example, the following creates a simulation named ``my_simulation_name`` with a duration of 1 second:

.. tab-set-code::

    .. code-block:: json

        "simulations": {
            "my_simulation_name": {
              "simulator": "neuron",
              "duration": 1000,
        }

    .. code-block:: python


        config.simulations.add("my_simulation_name",
          simulator="neuron",
          duration=1000,
        )


