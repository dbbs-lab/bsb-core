########
Overview
########

The topology module helps the placement module determine the shape and
organization of the simulated space. Every simulated space contains a flat
collection of :class:`Partitions <.topology.Partition>` organized into a hierarchy
by a tree of :class:`Regions <.topology.Region>`.

Partitions are defined by a least dominant corner (e.g. ``(50, 50, 50)``) and a
most dominant corner (e.g. ``(90, 90, 90)``) referred to as the LDC and MDC
respectively. With that information the outer bounds of the partition are
defined. Partitions have to be able to determine a ``volume``, ``surface`` and
``voxels`` given some bounds to intersect with. On top of that they have to be
able to return a list of ``chunks`` they belong to given a chunk size.
