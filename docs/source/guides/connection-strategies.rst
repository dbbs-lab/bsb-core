#############################
List of connection strategies
#############################

Connection strategies starting whose name start with :class:`Connectome` are made for a
specific connection between 2 cell types, those that do not can be used for connections
between any cell type.

Shared configuration attributes
-------------------------------

* ``class``: A string that specifies which connection strategy to apply to the connection
  type.
* ``from_cell_types``: An array of objects with a ``type`` key indicating presynaptic
  cell types and optionally a ``compartments`` key for an array of compartment types::

    "from_cell_types": [
      {"type": "basket_cell", "compartments": ["axon"]},
      {"type": "stellate_cell", "compartments": ["axon"]}
    ]

* ``to_cell_types``: Same as ``from_cell_types`` but for the postsynaptic cell type.

:class:`VoxelIntersection <.connectivity.VoxelIntersection>`
=====================================================================

This strategy voxelizes morphologies into collections of cubes, thereby reducing the
spatial specificity of the provided traced morphologies by grouping multiple compartments
into larger cubic voxels. Intersections are found not between the seperate compartments
but between the voxels and random compartments of matching voxels are connected to eachother.
This means that the connections that are made are less specific to the exact morphology
and can be very useful when only 1 or a few morphologies are available to represent each
cell type.

* ``affinity``: A fraction between 1 and 0 which indicates the tendency of cells to form
  connections with other cells with whom their voxels intersect. This can be used to
  downregulate the amount of cells that any cell connects with.
* ``contacts``: A number or distribution determining the amount of synaptic contacts one
  cell will form on another after they have selected eachother as connection partners.

.. note::
  The affinity only affects the number of cells that are contacted, not the number of
  synaptic contacts formed with each cell.

:class:`FiberIntersection <.connectivity.FiberIntersection>`
=====================================================================

This strategy is a special case of `VoxelIntersection` that can be applied to morphologies
with long straight compartments that would yield incorrect results when approximated with
cubic voxels like in VoxelIntersection (e.g. Ascending Axons or Parallel Fibers in Granule
Cells). The fiber, organized into hierarchical branches, is split into segments, based on
original compartments length and configured resolution. Then, each branch is voxelized
into parallelepipeds: each one is built as the minimal volume with sides parallel to the
main reference frame axes, surrounding each segment. Intersections with postsynaptic
voxelized morphologies are then obtained applying the same method as in
`VoxelIntersection`.

* ``resolution``: the maximum length [um] of a fiber segment to be used in the fiber
  voxelization. If the resolution is lower than a compartment length, the compartment is
  interpolated into smaller segments, to achieve the desired resolution. This property
  impacts on voxelization of fibers not parallel to the main reference frame axes. Default
  value is 20.0 um, i.e. the length of each compartment in Granule cell Parallel fibers.
* ``affinity``: A fraction between 1 and 0 which indicates the tendency of cells to form
  connections with other cells with whom their voxels intersect. This can be used to
  downregulate the amount of cells that any cell connects with. Default value is 1.
* ``transform``: A set of attributes defining the transformation class for fibers that
  should be rotated or bended. Specifically, the `QuiverTransform` allows to bend fiber
  segments based on a vector field in a voxelized volume. The attributes to be set are:

  * ``quivers``: the vector field array
  * ``vol_res``: the size [um] of voxels in the volume where the vector field is defined.
    Default value is 25.0, i.e. the voxel size in the Allen Brain Atlas.
  * ``shared``: if the same transformation should be applied to all fibers or not

:class:`TouchingConvergenceDivergence <.connectivity.TouchingConvergenceDivergence>`
====================================================================================

* ``divergence``: Preferred amount of connections starting from 1 from_cell
* ``convergence``: Preferred amount of connections ending on 1 to_cell

:class:`ConnectomeGlomerulusGranule <.connectivity.ConnectomeGlomerulusGranule>`
================================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the dendrite length configured in the granule cell morphology.

:class:`ConnectomeGlomerulusGolgi <.connectivity.ConnectomeGlomerulusGolgi>`
============================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the dendrite radius configured in the Golgi cell morphology.

:class:`ConnectomeGolgiGlomerulus <.connectivity.ConnectomeGolgiGlomerulus>`
============================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the ``axon_x``, ``axon_y``, ``axon_z`` from the Golgi cell morphology
to intersect a parallelopipid Golgi axonal region with the glomeruli.

:class:`ConnectomeGranuleGolgi <.connectivity.ConnectomeGranuleGolgi>`
======================================================================

Creates 2 connectivity sets by default *ascending_axon_to_golgi* and
*parallel_fiber_to_golgi* but these can be overwritten by providing ``tag_aa``
and/or ``tag_pf`` respectively.

Calculates the distance in the XZ plane between granule cells and Golgi cells and
uses the Golgi cell morphology's dendrite radius to decide on the intersection.

Also creates an ascending axon height for each granule cell.

* ``aa_convergence``: Preferred amount of ascending axon synapses on 1 Golgi cell.
* ``pf_convergence``: Preferred amount of parallel fiber synapses on 1 Golgi cell.

:class:`ConnectomeGolgiGranule <.connectivity.ConnectomeGolgiGranule>`
======================================================================

No configuration, it connects each Golgi to each granule cell that it shares a
connected glomerules with.

:class:`ConnectomeAscAxonPurkinje <.connectivity.ConnectomeAscAxonPurkinje>`
============================================================================

Intersects the rectangular extension of the Purkinje dendritic tree with the granule
cells in the XZ plane, uses the Purkinje cell's placement attributes ``extension_x``
and ``extension_z``.

* ``extension_x``: Extension of the dendritic tree in the X plane
* ``extension_z``: Extension of the dendritic tree in the Z plane

:class:`ConnectomePFPurkinje <.connectivity.ConnectomePFPurkinje>`
==================================================================

No configuration. Uses the Purkinje cell's placement attribute ``extension_x``.
Intersects Purkinje cell dendritic tree extension along the x axis with the x position
of the granule cells, as the length of a parallel fiber far exceeds the simulation
volume.
