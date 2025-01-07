.. _get-started:

===============
Top Level Guide
===============

The Brain Scaffold Builder is a **framework** that allows you to create workflows to
reconstruct and simulate neural network models.

.. figure:: /images/workflow.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/workflow_dark.png
  :figwidth: 90%
  :figclass: only-dark

A typical workflow of the BSB works as follows:

1. The user provides a configuration that describes the network they want to reconstruct
   and simulate.

   It is interpreted by the BSB as a series of tasks to perform.
2. The BSB creates the topology of the network (i.e., its shape and size).
3. The BSB places cells within the network, following a defined strategy.
4. The BSB connects the aforementioned cells according to connectivity rules.
5. The final circuit is simulated with an external simulator, with the BSB serving as an
   intermediate to deal with the model.

For each step of this workflow, the BSB also provides a list of predefined strategies and
tools to create your own with as little difficulty as possible. Hence, for most of your
reconstructions, you may not have to code!

The BSB acts here as a **black box** with an extended interface to allow you to properly
parametrize it.

Scaffold
========

.. figure:: /images/bsb_toplevel.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/bsb_toplevel_dark.png
  :figwidth: 90%
  :figclass: only-dark

The Brain **Scaffold** Builder revolves around the :doc:`Scaffold </core/scaffold>` object. A
scaffold ties together all the information in the :doc:`Configuration </config/files>` with the
:doc:`Storage </core/storage>`. The configuration contains your model description, while the
storage contains your model data, like concrete cell positions or connections.

Using the scaffold object, one can turn the abstract model configuration into a concrete
storage object full of neuroscience.

To do so, the configuration leverages configurable objects to describe the underlying neural network,
called **components**. Components define which methods and parameters should be used to reconstruct and
simulate the network. The ones that you will probably employ the most are:

* :guilabel:`Topology` defines the shape and volume of your network,
  (it is composed of :guilabel:`Regions` and :guilabel:`Partitions`),
* :guilabel:`Cell Types` allows you to estimate the cellular composition
  (and attach :guilabel:`Morphologies` when needed),
* :guilabel:`Placement` places cells,
* :guilabel:`Connectivity` connect cells,
* :guilabel:`Simulation` simulates the resulting network.

Assembled together these components form a linear workflow that will build your network from scratch.

| Through this interface, lies the ultimate goal of the entire framework:
| To let you explicitly define every component that is a part of your model, and all its related parameters,
  in such a way that a single CLI command, ``bsb compile``, can turn your configuration into a reconstructed
  biophysically detailed large scale neural network.

.. _config:

Configuration
=============

The ``Configuration`` object is organized as a hierarchical tree.
From the root, the main blocks branch off, consisting of nine main components:

* :guilabel:`network`
* :guilabel:`storage`
* :guilabel:`regions`
* :guilabel:`partitions`
* :guilabel:`morphologies`
* :guilabel:`cell_types`
* :guilabel:`placement`
* :guilabel:`connectivity`
* :guilabel:`simulation`

These blocks contain nested sub-blocks that form the network.
Additionally, there are two optional components: :guilabel:`after_placement` and :guilabel:`after_connectivity`,
where users can define specific hooks to run within the workflow.
All these components will be described in more detail in the following sections.

.. figure:: /images/configuration.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/configuration_dark.png
  :figwidth: 90%
  :figclass: only-dark

The configuration object contains only the description of the model, not its implementation (python code)
nor its data (stored in the storage object).
It can therefore be stored in a separate file (usually Json or Yaml) that can be easily interpreted by the BSB.

What is next?
=============
We are now going to introduce the different components through a tutorial, explaining how to build
:doc:`your first network <getting-started_reconstruction>` .
