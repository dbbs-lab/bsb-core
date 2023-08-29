##################
List of strategies
##################

:class:`VoxelIntersection <.connectivity.detailed.voxel_intersection.VoxelIntersection>`
========================================================================================

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

:class:`FiberIntersection <.connectivity.detailed.fiber_intersection.FiberIntersection>`
========================================================================================

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
* ``to_plot``: a list of cell fiber numbers (e.g. 0 for the first cell of the presynaptic
  type) that will be plotted during connection creation using `plot_fiber_morphology`.
* ``transform``: A set of attributes defining the transformation class for fibers that
  should be rotated or bended. Specifically, the `QuiverTransform` allows to bend fiber
  segments based on a vector field in a voxelized volume. The attributes to be set are:

  * ``quivers``: the vector field array, of shape e.g. ``(3, 500, 400, 200))`` for
    a volume with 500, 400 and 200 voxels in x, y and z directions, respectively.
  * ``vol_res``: the size [um] of voxels in the volume where the quiver field is defined.
    Default value is 25.0, i.e. the voxel size in the Allen Brain Atlas.
  * ``vol_start``: the origin of the quiver field volume in the reconstructed volume reference frame.
  * ``shared``: if the same transformation should be applied to all fibers or not
