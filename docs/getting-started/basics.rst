=======================
Managing Scaffold files
=======================

.. note::

    This guide is a continuation of the
    :doc:`Getting Started guide </getting-started/getting-started_reconstruction>`.

In this tutorial, we assume that you have successfully reconstructed a network with BSB.
In this guide, you will learn how to access the data stored in your produced network

Loading a scaffold from file
============================

You can load a stored network from file using :meth:`from_storage <bsb.core.from_storage>`:

.. literalinclude:: /../examples/tutorials/load_data.py
   :language: python
   :lines: 3-5
   :emphasize-lines: 3

Once you have loaded the `Scaffold` object, you have access to its `Configuration`
and `Storage`.

.. tip::

    Remember that the storage is filled with the data produced during the reconstruction
    while the configuration describes the process to obtain the data
    (read again :doc:`this section <top-level-guide>` if it is not clear to you).

Accessing Scaffold data
=======================

Configuration
-------------

The Configuration of a Scaffold is available as ``scaffold.configuration``.
Its root components such as ``cell_types``, ``placement`` and others are
also directly available in the Scaffold object, so you can avoid some
needless typing and repetition.

.. literalinclude:: /../examples/tutorials/load_data.py
   :language: python
   :lines: 8-11
   :emphasize-lines: 1

Placement data
--------------

The placement data is available through the :class:`PlacementSet <.storage.interfaces.PlacementSet>`
interface. You can access stored placement sets through their name or their cell type.
This example shows how to access the cell positions of each population:

.. literalinclude:: /../examples/tutorials/load_data.py
   :language: python
   :lines: 14-19
   :emphasize-lines: 2

Take some time to familiarize yourself with `PlacementSet` methods
:doc:`here </placement/placement-set>`.

Connectivity data
-----------------

The connectivity data is available through the
:class:`ConnectivitySet <.storage.interfaces.ConnectivitySet>` interface.
Remember that connection sets are labelled by default according to the connection strategy
used to obtain them (and the pre and postsynaptic cell types in case their are more than one).

Here we are going to retrieve one connection set using its name (``"my_CS"``) and print the neuron
id of each connected pair.

.. literalinclude:: /../examples/tutorials/load_data.py
   :language: python
   :lines: 23-25
   :emphasize-lines: 1

See more info on how to manipulate `ConnectivitySet` :doc:`here </connectivity/connectivity-set>`.

What is next?
=============
Learn how to :doc:`assign morphologies to your neurons <include_morphos>` or check out this
tutorial on :doc:`how to run a simulation <guide_simulation>` with your network.

