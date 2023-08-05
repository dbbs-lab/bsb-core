##############
Placement sets
##############

:class:`PlacementSets <.storage.interfaces.PlacementSet>` are constructed from the
:class:`~.storage.Storage` and can be used to retrieve the positions, morphologies,
rotations and additional datasets.

.. note::

  Loading datasets from storage is an expensive operation. Store a local reference to the
  data you retrieve instead of making multiple calls.

Retrieving a PlacementSet
=========================

Multiple ``get_placement_set`` methods exist in several places as shortcuts to create the
same :class:`~.storage.interfaces.PlacementSet`. If the placement set does not exist, a
``DatesetNotFoundError`` is thrown.

.. code-block:: python

  from bsb.core import from_storage

  network = from_storage("my_network.hdf5")
  ps = network.get_placement_set("my_cell")
  # Alternatives to obtain the same placement set:
  ps = network.get_placement_set(network.cell_types.my_cell)
  ps = network.cell_types.my_cell.get_placement_set()
  ps = network.storage.get_placement_set(network.cell_types.my_cell)


Identifiers
===========

Cells have no global identifiers, instead you use the indices of their data, i.e. the
n-th position belongs to cell n, and so will the n-th rotation.

Positions
=========

The positions of the cells can be retrieved using the
:meth:`~.storage.interfaces.PlacementSet.load_positions` method.

.. code-block:: python

  for n, position in enumerate(ps.positions):
    print("I am", ps.tag, "number", n)
    print("My position is", position)

Morphologies
============

The positions of the cells can be retrieved using the
:meth:`~.storage.interfaces.PlacementSet.load_morphologies` method.

.. code-block:: python

  for n, (pos, morpho) in enumerate(zip(ps.load_positions(), ps.load_morphologies())):
    print("I am", ps.tag, "number", n)
    print("My position is", position)

.. warning::

	Loading morphologies is especially expensive.

  :meth:`~.storage.interfaces.PlacementSet.load_morphologies` returns a
  :class:`~.morphologies.MorphologySet`. There are better ways to iterate over it using
  either **soft caching** or **hard caching**.

Rotations
=========

The positions of the cells can be retrieved using the
:meth:`~.storage.interfaces.PlacementSet.load_rotations` method.

Additional datasets
===================

Not implemented yet.
