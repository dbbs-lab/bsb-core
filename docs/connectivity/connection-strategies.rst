##################
List of strategies
##################

:class:`AllToAll <.connectivity.general.AllToAll>`
==================================================

This strategy connects each presynaptic neuron to all the postsynaptic neurons.
It therefore creates one connection for each unique pair of neuron.

:class:`FixedIndegree <.connectivity.general.FixedIndegree>`
============================================================

This strategy connects to each postsynaptic neuron, a fixed number of uniform randomly selected
presynaptic neurons.

* ``indegree``: Number of neuron to connect for each postsynaptic neuron.

:class:`FixedOutdegree <.connectivity.general.FixedOutdegree>`
==============================================================

This strategy connects to each presynaptic neuron, a fixed number of uniform randomly selected
postsynaptic neurons.

* ``outdegree``: Number of neuron to connect for each presynaptic neuron.

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