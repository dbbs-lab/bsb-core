==============
Cell Placement
==============

Cell placement is handled by the :doc:`placement module </scaffold/placement>`.
This module will place the cell types in the layers based on a certain
:ref:`placement_strategy`.

.. contents::

Introduction
------------

Placement occurs as the first step during network architecture compilation.

The placement order happens in a lowest cell count first fashion unless specified
otherwise in the cell type's :ref:`placement_strategy`
configuration.


.. _placement_strategy:

Placement Strategy Interface
----------------------------

Each cell type has to specify a placement strategy that determines the algorithm
used to place cells. The most common placement strategy in the scaffold is the
:class:`LayeredRandomWalk`. The placement strategy is an interface that
will call a ``place`` method when placement occurs. The place method gets the cell type
that's being placed as a parameter.
