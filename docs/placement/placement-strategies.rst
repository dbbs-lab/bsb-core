############################
List of placement strategies
############################

ParticlePlacement
*****************

*Class*: :class:`bsb.placement.ParticlePlacement <.placement.particle.ParticlePlacement>`

This class considers the cells as spheres and
bumps them around as repelling particles until there is no overlap between them.

RandomPlacement
*****************

*Class*: :class:`bsb.placement.RandomPlacement <.placement.particle.RandomPlacement>`

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

SatellitePlacement
******************

*Class*: :class:`bsb.placement.Satellite
*<.placement.satellite.Satellite>`

This class place new cells as satellites of an existing one (the planet). Place the new cell in the neighbourhood
of the associated cell at a random distance, depending of the radius of the cells.

It is necessary to specify the Cell Types of cells to use as planets with the attribute ``planet_types``,
it is also possible to choose the number of satellites per planet with ``per_planet`` attribute.

.. code-block:: json

  {
    "cell_types": {
      "golgi_cell": {
        "placement": {
          "class": "bsb.placement.satellite.Satellite",
          "layer": "granular_layer",
          "planet_types": ["my_cell_type"],
          "per_planet": 2
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


