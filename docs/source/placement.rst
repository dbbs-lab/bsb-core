##############
Cell Placement
##############

Cell placement is handled by the :doc:`placement module </scaffold/placement>`.
This module will place the cell types in the layers based on a certain
:ref:`placement_strategy`.

Placement occurs as the first step during network architecture compilation.

The placement order happens in a lowest cell count first fashion unless specified
otherwise in the cell type's :ref:`placement_strategy` configuration.

.. contents::

*************
Configuration
*************

.. _specifying_cell_count:

Specifying cell count
=====================

Specifying cell count can be done with ``count``, ``density`` (µm^-3),
``planar_density`` (µm^-2) or a ratio to another cell with
``placement_relative_to`` (other cell type) and either ``density_ratio`` to
place with their density multiplied by the given ratio or
``placement_count_ratio`` to place with their count multiplied by the given
ratio

.. _placement_strategy:

****************************
Placement Strategy Interface
****************************

Each cell type has to specify a placement strategy that determines the algorithm
used to place cells. The most common placement strategy in the scaffold is the
:class:`LayeredRandomWalk <.placement.LayeredRandomWalk>`. The placement strategy is an interface that
will call a ``place`` method when placement occurs. The place method gets the cell type
that's being placed as a parameter.
