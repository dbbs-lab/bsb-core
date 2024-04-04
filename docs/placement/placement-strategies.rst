############################
List of placement strategies
############################

RandomPlacement
*****************

*Class*: :class:`bsb.placement.RandomPlacement <.placement.random.RandomPlacement>`


ParallelArrayPlacement
**********************

*Class*: :class:`bsb.placement.ParallelArrayPlacement
<.placement.arrays.ParallelArrayPlacement>`

FixedPositions
**************

*Class*: :class:`bsb.placement.FixedPositions <.placement.strategy.FixedPositions>`

This class places the cells in fixed positions specified in the attribute ``positions``.

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
