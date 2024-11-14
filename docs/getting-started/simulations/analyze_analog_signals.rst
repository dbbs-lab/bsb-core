.. _guide_analyze_analog:

########################################
Analyze multi-compartment neuron results
########################################

.. note::

    This guide is a continuation of the
    :doc:`Simulation guide <guide_neuron>`.

After a simulation, BSB will write its outcomes in ``.nio`` files. These files leverage the HDF5
(Hierarchical Data Format version 5) format, which is widely employed in scientific computing,
engineering, and data analysis. HDF5 files store data in groups (or blocks), in a dictionary
fashion, which allows for efficient management and organization of large datasets.

The content is read/written using the :doc:`Neo Python package <neo:index>`, a library designed
for handling electrophysiology data.

In our case, we have only one ``block`` of data since we only ran one simulation. We also have a
unique ``segment`` since all our data is recorded with the same time frame (see
:doc:`Neo docs <neo:read_and_analyze>` for more details).
Let's extract the simulation block data produced by your last simulation. First, load the content from
your ``simulation-results/NAME_OF_YOUR_NEO_FILE.nio`` file, use the following code:

.. literalinclude:: /../examples/tutorials/analyze_analog_results.py
    :language: python
    :lines: 1-7

If you followed the previous simulation example, the :guilabel:`analogsignals` attribute in the block
should contain a list of all measured signals: the membrane potential recorded by the
:guilabel:`vrecorder` device and the synapse current obtained from the :guilabel:`synapses_rec` device.

Each :class:`AnalogSignal <neo.core.AnalogSignal>` object contains information about the device name,
the sampling rate, and an array of the simulated measurement values.
Additional information is available through the annotations attribute.

.. literalinclude:: /../examples/tutorials/analyze_analog_results.py
    :language: python
    :lines: 10-49

This code generates one plot for each analog signal recorded, meaning one plot per device for each
cell or synapse recorded. The resulting figures are saved in the ``simulation-results`` folder.

Here are some example of the figures that are produced:

.. figure:: /getting-started/data/vrecorder_example.png
  :figwidth: 90%

  Example of the membrane potential recorded for the cell with 0 ID.

.. figure:: /getting-started/data/synapse_recorder_example.png
  :figwidth: 90%

  Example of the AMPA synapse current recoded for the cell with 2 ID.

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1


    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Make custom components
       :link: guide_components
       :link-type: ref

       Learn how to write your own components to e.g. place or connect cells.

    .. grid-item-card:: :octicon:`repo-clone;1em;sd-text-warning` Command-Line Interface
        :link: cli-guide
        :link-type: ref

        Familiarize yourself with BSB's CLI.

    .. grid-item-card:: :octicon:`gear;1em;sd-text-warning` Learn about components
       :link: main-components
       :link-type: ref

       Explore more about the main components.

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
        :link: examples
        :link-type: ref

        Explore more advanced examples
