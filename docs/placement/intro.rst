==========
Placement
==========
This is the block that fills cells into partitions.
The main object is the PlacementStrategy, a set of instructions for the definition of the positions of each cells in the partition volume.

The BSB offers some strategies out of the box (here is a :doc:`list </placement/placement-strategies>`), or you can implement your own.

The data is stored in :doc:`PlacementSets </placement/placement-set>` per cell type.