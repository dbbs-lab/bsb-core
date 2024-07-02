#######
Regions
#######

In BSB a regions is a hierarchical container of :doc:`partitions <topology/partitions>`
used to manage different areas of the network.

.. _stack-region:

=======================
List of builtin regions
=======================

BSB provide an implemented :class:`stack <.topology.regions.Stack>` type region that
allow to create a pile of partitions.