===============
Top Level Guide
===============

.. figure:: /images/bsb_toplevel.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/bsb_toplevel_dark.png
  :figwidth: 90%
  :figclass: only-dark

The Brain **Scaffold** Builder revolves around the :class:`~.core.Scaffold` object. A
scaffold ties together all the information in the :class:`~.config.Configuration` with the
:class:`~.storage.Storage`. The configuration contains your entire model description,
while the storage contains your model data, like concrete cell positions or connections.

Using the scaffold object one can turn the abstract model configuration into a concrete
storage object full of neuroscience. For it to do so, the configuration needs to describe
which steps to take to place cells, called ``Placement``, which steps to take to connect
cells, called ``Connectivity``, and what representations to use during ``Simulation`` for
those cells and connections. All of these configurable objects can be accessed from the
scaffold object. Placement under ``scaffold.placement``, etc etc...

Also, using the scaffold object, you can inspect the data in the storage by using the
:class:`~.storage.interfaces.PlacementSet` and
:class:`~.storage.interfaces.ConnectivitySet` APIs. PlacementSets can be obtained with
:meth:`scaffold.get_placement_set <.core.Scaffold.get_placement_set>`, ConnectivitySets
with :meth:`scaffold.get_connectivity_set <.core.Scaffold.get_placement_set>` etc etc...

The configuration object contains a structured tree of configurable objects, that in
totality describe your network model. You can either fill out configuration file to be
parsed, or write the objects yourself in Python. There are several parts of a
configuration to be filled out, take a look at `config`_ to find out the details.

The storage object provides access to an underlying engine that performs read and write
operations in a certain data format. You can use the storage object to manipulate the data
in your model, but usually it's better if the scaffold object is allowed to translate
configuration directly into data, so that anyone can take a look at the config and know
exactly what data is in storage, and how it got there!

Ultimately this is the goal of the entire framework: To let you explicitly define every
component that is a part of your model, and all its parameters, in such a way that a
single CLI command, ``bsb compile``, can turn your configuration into a reconstructed
biophysically detailed large scale neural network, with all its parameters explicitly
presented to any reader in a human readable configuration file.

Workflow
========

.. figure:: /images/workflow.png
  :figwidth: 90%
  :figclass: only-light

.. figure:: /images/workflow_dark.png
  :figwidth: 90%
  :figclass: only-dark

The framework promotes iterative improvements on the model. Start small, and incrementally
add on every piece you need after validating the last!

.. _config:

Configuration
=============
