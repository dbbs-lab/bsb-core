############
Introduction
############

A configuration file describes a scaffold model. It contains the instructions to place and
connect neurons, how to represent the cells and connections as models in simulators and
what to stimulate and record in simulations.

The default configuration format is JSON and a standard configuration file might look like
this:

.. code-block:: json

  {
    "storage": {
      "engine": "hdf5",
      "root": "my_network.hdf5"
    },
    "network": {
      "x": 200,
      "z": 200
    },
    "regions": {

    },
    "partitions": {

    },
    "cell_types": {

    },
    "connection_types": {

    }
  }

The ``regions``, ``layers````cell_types`` and ``connection_types`` spaceholders would hold
configuration for :class:`Regions <.objects.Region>`, :class:`Layers <.objects.Layer>`,
:class:`CellTypes <.objects.CellType>` and :class:`ConnectionStrategies
<.connectivity.ConnectionStrategy>` respectively.

Network
#######

This node contains some basic properties of the network configured in this file.

The :guilabel:`x` and :guilabel:`z` attributes are loose indicators of the scale of the
simulation. You can use them to scale the volume of your layers or for other mechanisms
that determine the region in which to place your cells. They do not restrict placement,
things can still be placed outside the specified [0, x] and [0, z] region.

Basic cell types
################

For a more complete guide see :doc:`/config/cell-types`

Spatial
-------

For a more complete guide see :doc:`/config/spatial`

Placement
---------

For a more complete guide see :doc:`/config/placement-strategies`

Basic layers
############

For a more complete guide see :doc:`/config/layers`

Basic regions
#############

For a more complete guide see :doc:`/config/regions`

Basic connection strategies
###########################

For a more complete guide see :doc:`/config/connection-strategies`

Postprocessing hooks
####################

For a more complete guide see :doc:`/config/postprocessing`

After placement
---------------

After connectivity
------------------
