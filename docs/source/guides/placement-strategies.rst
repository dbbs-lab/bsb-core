############################
List of placement strategies
############################

*****************
PlacementStrategy
*****************

Configuration
=============

* ``layer``: The layer in which to place the cells.
* ``soma_radius``: The radius in µm of the cell body.
* ``count``: Determines cell count absolutely.
* ``density``: Determines cell count by multiplying it by the placement volume.
* ``planar_density``: Determines cell count by multiplying it by the placement surface.
* ``placement_relative_to``: The cell type to relate this placement count to.
* ``density_ratio``: A ratio that can be specified along with ``placement_relative_to``
  to multiply another cell type's density with.
* ``placement_count_ratio``: A ratio that can be specified along with
  ``placement_relative_to`` to multiply another cell type's placement count with.

**********************
ParallelArrayPlacement
**********************

*Class*: :class:`.placement.ParallelArrayPlacement`

**************
FixedPositions
**************

*Class*: :class:`.placement.FixedPositions`

This class places the cells in fixed positions specified in the attribute ``positions``.

Configuration
=============

* ``positions``: a list of 3D points where the neurons should be placed. For example:

.. code-block:: json

  {
    "cell_types": {
      "golgi_cell": {
        "placement": {
          "class": "bsb.placement.FixedPositions",
          "layer": "granular_layer",
          "count": 1,
          "positions": [[40.0,0.0,-50.0]]
          }
      },
    }
  }
