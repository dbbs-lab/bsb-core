############################
List of placement strategies
############################

RandomPlacement
*****************

*Class*: :class:`bsb.placement.RandomPlacement <.placement.random.RandomPlacement>`

This class place cells in random postition without caring about overlaps. Here is an example with 10 cells.

.. code-block:: json

  {
    "cell_types": {
      "golgi_cell": {
        "placement": {
          "class": "bsb.placement.particle.RandomPlacement",
          "layer": "granular_layer",
          "count": 10
          }
      },
    }

ParallelArrayPlacement
**********************

*Class*: :class:`bsb.placement.ParallelArrayPlacement
<.placement.arrays.ParallelArrayPlacement>`

This class place cells in an aligned array, it create a lattice with fixed spacing and with the desired angle.
It is necessary to specify ``spacing_x`` and ``angle`` attributes.

.. code-block:: json

  {
    "cell_types": {
      "golgi_cell": {
        "placement": {
          "class": "bsb.placement.arrays.ParallelArrayPlacement",
          "layer": "granular_layer",
          "count": 100,
          "spacing_x": 10,
          "angle": 0
          }
      },
    }
  }

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


