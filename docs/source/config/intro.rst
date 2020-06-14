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
    "layers": {

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
