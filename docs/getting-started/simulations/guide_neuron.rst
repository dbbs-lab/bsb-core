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
      :lines: 1-71,140

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
      :lines: 72-78

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 56-65

Cell Models
-----------




