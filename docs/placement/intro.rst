==========
Placement
==========
This block is responsible for placing cells into partitions. The main object is the PlacementStrategy,
which provides a set of instructions for defining the positions of each cell within the partition volume.
The BSB offers several built-in strategies (here is a :doc:`list </placement/placement-strategies>`),
or you can implement your own.
The placement data is stored in :doc:`PlacementSets </placement/placement-set>` for each cell type.