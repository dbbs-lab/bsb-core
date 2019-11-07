##############
Cell Placement
##############

Cell placement is handled by the :doc:`placement module </scaffold/placement>`.
This module will place the cell types in the layers based on a certain
:ref:`placement_strategy`.

Placement occurs as the first step during network architecture compilation.

The placement order starts from cell type with the lowest cell count first
unless specified otherwise in the cell type's placement configuration.

See the :doc:`/configuration/placement-strategies`

.. contents::

*************
Configuration
*************

.. _specifying_cell_count:

Cell count
=====================

Specifying cell count can be done with ``count``, ``density`` (µm^-3),
``planar_density`` (µm^-2) or a ratio to another cell with
``placement_relative_to`` (other cell type) and either ``density_ratio`` to
place with their density multiplied by the given ratio or
``placement_count_ratio`` to place with their count multiplied by the given
ratio

.. _specifying_placement_order:

Placement order
================

By default the cell types are placed sorted from least to most cells per type.
This default order can be influenced by specifying an ``after`` attribute
in the cell type's placement configuration. This is an array of cell type names
which need to be placed before this cell type.

.. _cell_epsilon:

Cell epsilon
============
The :class:`.placement.LayeredRandomWalk` calculates the total volume of the layer
and divides that by the amount of cells to place, to yield a per cell volume
(not taking into account sphere packing).
The sphere of that volume is calculated and taken as the per cell radius.
Subtracting the cell type radius yields the cell epsilon. This is the amount of
"wiggle room" each cell has in its assigned placement volume. This epsilon is
also used to determine how far to place cells from eachother.

.. _placement_strategy:

******************
Placement Strategy
******************

Each cell type has to specify a placement strategy that determines the algorithm
used to place cells. The most common placement strategy in the scaffold is the
:class:`LayeredRandomWalk <.placement.LayeredRandomWalk>`.
The placement strategy is an interface whose ``place`` method is called when
placement occurs.

Placing cells
=============

Call the scaffold instance's :func:`.scaffold.Scaffold.place_cells` function to
place cells in the simulation volume.
