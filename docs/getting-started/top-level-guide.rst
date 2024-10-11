    .. _get-started:

===============
Top Level Guide
===============

.. figure:: /images/bsb_toplevel.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/bsb_toplevel_dark.png
  :figwidth: 90%
  :figclass: only-dark

The Brain **Scaffold** Builder revolves around the :doc:`Scaffold </components/scaffold>` object. A
scaffold ties together all the information in the ``Configuration`` with the
:doc:`Storage </components/storage>`. The configuration contains your model description, while the
storage contains your model data, like concrete cell positions or connections.

Using the scaffold object one can turn the abstract model configuration into a concrete
storage object full of neuroscience. For it to do so, the configuration needs to describe
which steps to take to place cells, called ``Placement``, which steps to take to connect
cells, called ``Connectivity``, and what representations to use during ``Simulation`` for
those cells and connections. All of these configurable objects can be accessed from the
scaffold object, under ``network.placement``, ``network.connectivity``,
``network.simulations``, ...


Ultimately this is the goal of the entire framework: To let you explicitly define every
component and parameter that is a part of your model, and all its parameters, in such a
way that a single CLI command, ``bsb compile``, can turn your configuration into a
reconstructed biophysically detailed large scale neural network.

Workflow
========

.. figure:: /images/workflow.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/workflow_dark.png
  :figwidth: 90%
  :figclass: only-dark

.. _config:

Configuration
=============

The ``Configuration`` object is organized as a hierarchical tree.
From the root, the main blocks branch off, consisting of nine required components: :guilabel:`network`,
:guilabel:`storage`, :guilabel:`regions`, :guilabel:`partitions`, :guilabel:`morphologies`, :guilabel:`cell types`, :guilabel:`placement`, :guilabel:`connectivity`, and :guilabel:`simulation`.
These blocks contain nested sub-blocks that form the network.
Additionally, there are two optional blocks: :guilabel:`after_placement` and :guilabel:`after_connectivity`, where users can define specific hooks to run within the workflow.

.. figure:: /images/configuration.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/configuration_dark.png
  :figwidth: 90%
  :figclass: only-dark

Configuration File
------------------

A configuration file describes the components of a scaffold model. It contains the
instructions to place and connect neurons, how to represent the cells and connections as
models in simulators and what to stimulate and record in simulations.

The default configuration format is JSON, but YAML is also supported.
A standard configuration file is structured as follows:

.. include:: _empty_config_example.rst


