############
Introduction
############

The topology module allows you to make abstract descriptions of the spatial layout of
pieces of the region you are modelling. :class:`Partitions
<.topology.partition.Partition>` help you define shapes to place into your region such as
layers, cubes, spheres, meshes and so on. :class:`Regions <.topology.region.Region>` help
you put those pieces together by arranging them on top of each other, next to each other,
away from each other, ... You can define your own ``Partitions`` and ``Regions``; as long
as each partition knows how to transform itself into a collection of voxels (volume
pixels) and each region knows how to arrange its children these elements can become the
building blocks of an arbitrarily large and parallelizable model description.
