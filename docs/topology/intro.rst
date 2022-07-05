############
Introduction
############

Layouts
=======

The topology module allows you to make abstract descriptions of the spatial layout of
pieces of the region you are modelling. :class:`Partitions
<.topology.partition.Partition>` define shapes such as layers, cubes, spheres, and meshes.
:class:`Regions <.topology.region.Region>` put partitions together by arranging them
hierarchically. The topology is formed as a tree of regions, that end downstream in a
terminal set of partitions.

To initiate the topology, the network size hint is passed to the root region, which
subdivides it for their children to make an initial attempt to lay themselves out. Once
handed back the initial layouts of their children, parent regions can propose
transformations to finalize the layout. If any required transformation proposals fail to
meet the configured constraints, the layout process fails.

Example
-------

.. figure:: /images/layout.png
  :figwidth: 350px
  :figclass: only-light
  :align: center

.. figure:: /images/layout_dark.png
  :figwidth: 350px
  :figclass: only-dark
  :align: center

The root :class:`Group <.topology.region.RegionGroup>` receives the network X, Y, and Z. A
``Group`` is an inert region and simply passes the network boundaries on to its children.
The :class:`~.topology.partition.Voxels` loads its voxels, and positions them absolutely,
ignoring the network boundaries. The :class:`~.topology.region.Stack` passes the volume on
to the :class:`Layers <.topology.partition.Layer>` who fill up the space and occupy their
thickness. They return their layout up to the parent ``Stack``, who in turn proposes
translations to the layers in order to stack them on top of the other. The end result is
stack beginning from the network starting corner, with 2 layers as large as the network,
with their respective thickness, and absolutely positioned voxels.
