################################
Run your first NEURON simulation
################################

.. note::

    This guide uses notions on the BSB reconstructions that are explained in
    :doc:`Getting Started guide </getting-started/getting-started_reconstruction>`.

In this tutorial, we present how to configure a NEURON simulation for a multi-compartment
neuron network.

Install requirements
====================

`NEURON <https://www.neuron.yale.edu/neuron/>`_ is one of the supported simulators of the
BSB. As for the other simulator, its adapter code is stored in a separate repository:
`bsb-neuron <https://github.com/dbbs-lab/bsb-neuron>`_

So, you would need to install it with pip:

.. code-block:: bash

    pip install bsb-neuron[parallel]

We will also need some model files for NEURON which you can obtain and link to bsb like so:

.. code-block:: bash

    pip install dbbs-catalogue

BSB reconstruction for this tutorial
====================================

For this example, we will build a network consisting of a single ``layer`` of
``stellate_cells`` connected through axon-dendrite overlap, using the strategy
:doc:`VoxelIntersection </connectivity/connection-strategies>`.
The morphology of a custom stellate cell is provided
:download:`here </getting-started/data/StellateCell.swc>`.
Please save this file in your project folder as ``StellateCell.swc``.

The network configuration should be as follows:

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 1-68

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 1-54

Copy the configuration in you favorite format and put it in the project folder
as ``neuron-simulation.json`` or  as ``neuron-simulation.py``

Then, the configuration should be compiled:

.. code-block:: bash

    bsb compile --verbosity 3 neuron-simulation.json
    # or
    python neuron-simulation.py

Now we have to configure the simulation block.

Configuration of the simulation
===============================

We want here to see the postsynaptic response of our cells upon receiving an
excitatory input. Each cell will receive one spike on their dendrites and
we will check its effect on the postsynaptic current.

Let's start by configuring the global simulation parameters: first of all,
define a :guilabel:`simulator`; in our example, we are setting it to
use NEURON.
Then you need to define the :guilabel:`resolution` (the time step of the simulation in
milliseconds), the :guilabel:`duration` (the total length of the simulation in
milliseconds) and the :guilabel:`temperature` (celsius unit).

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 69-74

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 56-61

Cell Models
-----------

For each **cell type** population in your network, you must assign a **NEURON model**
to define the cell's behavior.

In short, these models encapsulate all the specifications for ion channels and synapses
covering all compartments of the neuron. Discussing NEURON model characteristics is
beyond the scope of this guide; therefore, a ready-to-use Stellate model is provided
:download:`here </../examples/tutorials/Stellate.py>`. Save it as a ``Stellate.py``
file in your project folder and review its contents.

Within the model file, you will find a model definition called
:guilabel:`definitionStellate`, which includes all the customized parameters. This is
the object you will refer to in your configuration. Note also that the parameters for
the ion channel mechanisms are in the attribute :guilabel:`cable_types`.


.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 75-80

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 63-65

Connection Models
-----------------

For each connection type of your network, you also need to provide a NEURON model
describing its synapses' dynamics. Similar to the :guilabel:`cell_models` block, for
each :guilabel:`connection_model` you should use a key that corresponds to a
``ConnectivitySet`` created during reconstruction (as explained in the previous
:doc:`section </getting-started/getting-started_reconstruction>`).
In this example, to the :guilabel:`stellate_to_stellate` connection is assigned a
reference to one of the :guilabel:`synapse_types`, defined in the ``Stellate.py``
model file: :guilabel:`GABA`.

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 81-86

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 66-76

To each synapse is assigned a :guilabel:`weight` of 0.001 and a :guilabel:`delay` (ms) of 1.

Devices
-------

In the :guilabel:`devices` block, include all interfaces you wish to use for interacting with
the network.
These devices correspond typically to stimulators and measurement instruments.

Use the :guilabel:`device` key to select the type of device.
We also introduce here the :guilabel:`targetting` concept for the devices: This configuration
node allows you to filter elements of your neuron circuit to which you want to link your
devices (see the targetting section on :doc:`this page </simulation/intro>` for more details).

.. tab-set-code::

    .. literalinclude:: /getting-started/configs/guide-neuron.json
      :language: json
      :lines: 87-127

    .. literalinclude:: /../examples/tutorials/neuron-simulation.py
      :language: python
      :lines: 77-110

In this example, a :guilabel:`spike_generator` is used to produce ``1`` spike (attribute
:guilabel:`number`) at ``9`` ms and send it to the cell with ID ``0`` (using the
:guilabel:`targetting`) after ``1`` ms of delay and a :guilabel:`weight` of ``0.01``.
The stimulus targets the ``AMPA`` and ``NMDA`` (excitatory) synapses located on the ``dendrites``
of the cell.

The membrane potential is recorded using a :guilabel:`voltage_recorder`, which collects the
signal from within a ``100`` µm radius sphere at the center of the circuit. Hence, not all cells
might be recorded.

Synapse activity is monitored with a :guilabel:`synapse_recorder` for all the synaptic types on
the cell's dendrites, within the same spherical region. Here too, not all synapses might be recorded.

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

    bsb reconfigure my_network.hdf5 neuron-simulation.json

  .. code-block:: python

    storage = scaffold.storage
    storage.store_active_config(config)

Once this is done, create a folder in which to store your simulation results:

.. code-block:: bash

    mkdir simulation-results

You can now run your simulation:

.. tab-set-code::

  .. code-block:: bash

    bsb simulate my_network.hdf5 neuronsim -o simulation-results

  .. code-block:: python

        from bsb import from_storage

        scaffold = from_storage("my_network.hdf5")
        result = scaffold.run_simulation("neuronsim")
        result.write("simulation-results/neuronsimulation.nio", "ow")

The results of the simulation will be stored in the ``"simulation-results"`` folder.

.. note::
    If you run the simulation with the command line interface, the name of the output nio file is randomized by BSB.

For more detailed information about simulation modules,
please refer to the :doc:`simulation section </simulation/intro>`.

Congratulations, you simulated your first BSB reconstructed network with NEURON!

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1


    .. grid-item-card:: :octicon:`fold-up;1em;sd-text-warning` Analyze your Results
        :link: analyze_analog_signals
        :link-type: doc

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



