#######################
Configuration reference
#######################

==========
Root nodes
==========

.. include:: _empty_root_nodes.rst

Storage
=======

.. note::

	Storage nodes host plugins and can contain plugin-specific configuration.

.. include:: _full_storage_node.rst

* :guilabel:`engine`: The name of the storage engine to use.
* :guilabel:`root`: The storage engine specific identifier of the location of the storage.

Network
=======

.. include:: _full_network_node.rst

* :guilabel:`x`, :guilabel:`y` and :guilabel:`z`:
  Loose indicators of the scale of the network. They are handed to the topology of the
  network to scale itself. They do not restrict cell placement.
* :guilabel:`chunk_size`:
  The size used to parallelize the topology into multiple rhomboids. Can be a list of 3
  floats for a rhomboid or 1 float for cubes.

Regions
=======

.. note::

	Region nodes are components and can contain additional component-specific attributes.

.. include:: _full_region_node.rst

* :guilabel:`cls`: Class of the region.
* :guilabel:`offset`: Offset of this region to its parent in the topology.

Partitions
==========

.. note::

	Partition nodes are components and can contain additional component-specific attributes.

.. include:: _full_partition_node.rst

* :guilabel:`type`: Name of the partition component, or its class.
* :guilabel:`region`: By-name reference to a region.

Cell types
==========

.. include:: _full_cell_type_node.rst

* :guilabel:`entity`:
  Indicates whether this cell type is an abstract entity, or a regular cell.

* :guilabel:`spatial`: Node for spatial information about the cell.

  * :guilabel:`radius`: Radius of the indicative cell soma (μm).

  * :guilabel:`geometry`:
    Node for geometric information about the cell. This node may contain arbitrary keys
    and values, useful for cascading custom placement strategy attributes.

* :guilabel:`morphologies`: List of morphology selectors.

* :guilabel:`plotting`:

  * :guilabel:`display_name`: Name used for this cell type when plotting it.

  * :guilabel:`color`: Color used for the cell type when plotting it.

  * :guilabel:`opacity`: Opacity (non-transparency) of the :guilabel:`color`

Placement
=========

.. note::

	Placement nodes are components and can contain additional component-specific attributes.

.. include:: _full_placement_node.rst

* :guilabel:`cls`: Class name of the placement strategy algorithm to import.

* :guilabel:`cell_types`:
  List of cell type references. This list is used to gather placement indications for the
	underlying strategy. It is the underlying strategy that determines how they will
	interact, so check the component documentation. For most strategies, passing multiple
	cell types won't yield functional differences from having more cells in a single type.

* :guilabel:`partitions`:
  List of partitions to place the cell types in. Each strategy has their own way of
  dealing with partitions, but most will try to voxelize them (using
  :meth:`~.topology.partition.Partition.to_voxels`), and combine the voxelsets of each
	partition. When using multiple partitions, you can save memory if all partitions
	voxelize into regular same-size voxelsets.

* :guilabel:`overrides`:
  Cell types define their own placement indications in the :guilabel:`spatial` node, but
	they might differ depending on the location they appear in. For this reason, each
	placement strategy may override the information per cell type. Specify the name of the
	cell types as the key, and provide a dictionary as value. Each key in the dictionary
	will override the corresponding cell type key.

Connectivity
============

.. note::

	Connectivity nodes are components and can contain additional component-specific
	attributes.
