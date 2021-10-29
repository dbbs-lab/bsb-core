#######################
Configuration reference
#######################

==========
Root nodes
==========

.. include:: _empty_root_nodes.rst

Storage
=======

.. include:: _full_storage_node

* :guilabel:`engine`: The name of the storage engine to use.
* :guilabel:`root`: The storage engine specific identifier of the location of the storage.

.. note::

	Storage nodes are plugins and can contain plugin specific configuration.

Network
=======

.. include:: _full_network_node

* :guilabel:`x`, :guilabel:`y` and :guilabel:`z`: Loose indicators of the
scale of the network. They are handed to the topology of the network to scale itself. They
do not restrict cell placement.
* :guilabel:`chunk_size`: The size used to parallelize the topology into multiple rhomboids.

Regions
=======

.. include:: _full_region_node

* :guilabel:`cls`: Class of the region.
* :guilabel:`offset`: Offset of this region to its parent in the topology.

.. note::

	Region nodes are dynamic and can contain class specific configuration.

Partitions
==========

.. include:: _full_partition_node

* :guilabel:`cls`: Class of the partition.
* :guilabel:`region`: By-name reference to a region.

.. note::

	Partition nodes are dynamic and can contain class specific configuration.

Cell types
==========

.. include:: _full_cell_type_node
