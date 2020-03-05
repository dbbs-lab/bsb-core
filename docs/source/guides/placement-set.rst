##############
Placement sets
##############

:class:`PlacementSets <.models.PlacementSet>` are constructed from the
:doc:`/guides/output` and can be used to retrieve lists of identifiers, positions,
rotations and additional datasets. It can also be used to construct a list of
:class:`Cells <.models.Cell>` that combines that information into objects.

.. note::
  Loading these datasets from storage is an expensive operation. Store a local reference
  to the data you retrieve::

    data = placement_set.identifiers # Store a local variable
    cell0 = data[0] # NOT: placement_set.identifiers[0]
    cell1 = data[1] # NOT: placement_set.identifiers[1]

=========================
Retrieving a PlacementSet
=========================

The output formatter of the scaffold is responsible for retrieving the dataset from the
output storage. The scaffold itself has a method ``get_placement_set`` that takes a name
of a cell type as input which will defer to the output formatter and returns a
PlacementSet. If the placement set does not exist, an ``DatesetNotFoundError`` is thrown.

.. code-block:: python

  ps = scaffold.get_placement_set("granule_cell")


===========
Identifiers
===========

The identifiers of the cells of a cell type can be retrieved using the ``identifiers``
property. Identifiers are stored in a :ref:`format_nc_list`.

.. code-block:: python

  for n, cell_id in enumerate(ps.identifiers):
    print("I am", ps.tag, "number", n, "with ID", cell_id)

=========
Positions
=========

The positions of the cells can be retrieved using the ``positions`` property. This dataset
is not present on entity types:

.. code-block:: python

  for n, cell_id, position in zip(range(len(ps)), ps.identifiers, ps.positions):
    print("I am", ps.tag, "number", n, "with ID", cell_id)
    print("My position is", position)


=========
Rotations
=========

Some placement strategies or external data sources might also provide rotational information for each cell.
The ``rotations`` property works analogous to the ``positions`` property.

===================
Additional datasets
===================

Not implemented yet.
