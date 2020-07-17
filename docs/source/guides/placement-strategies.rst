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

*****************
LayeredRandomWalk
*****************

*Class*: :class:`.placement.LayeredRandomWalk`

The LayeredRandomWalk class will sudivide the layer volume into sublayers and divide
cells among them. In each sublayer a self-avoiding random walk will be performed
to distribute the cells throughout the sublayer. A random yitter is applied in
height and the sublayers are stacked on top of eachother.

Configuration
=============

* ``y_restriction`` *(optional)*: a 2 element array that can restrict placement
  to within certain y-axis bounds inside of a layer. For example ``y_restriction =
  [0.5, 1.0]`` will restrict placement to the top half of the layer.
* ``distance_multiplier_min/max``: These are multiplied by the :ref:`cell epsilon
  <cell_epsilon>` to
  obtain minimum and maximum distances between a newly placed and the previous
  cell. Default values are 0.75 and 1.25.

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
          "class": "scaffold.placement.FixedPositions",
          "layer": "granular_layer",
          "count": 1,
          "positions": [[40.0,0.0,-50.0]]
          }
      },
    }
  }
