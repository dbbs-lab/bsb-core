############################
List of placement strategies
############################

RandomPlacement
*****************

*Class*: :class:`bsb.placement.RandomPlacement <.placement.random.RandomPlacement>`

This class places cells in random positions without considering overlaps. Below is an example with 10 cells.

.. tab-set-code::

    .. code-block:: json

        "cell_types": {
            "my_cell": {
                "spatial": {
                    "count": 10,
                    "radius": 5
                }
            }
        },

        "placement": {
            "place_randomly":{
                "strategy": "bsb.placement.particle.RandomPlacement",
                "partitions": "my_layer",
                "cell_types": ["my_cell"]
            }
        },

    .. code-block:: python

      config.cell_types.add(
        "my_cell",
        spatial=dict(radius=5, count=10)
      )
      config.placement.add(
        "place_randomly",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["my_cell"],
      )
.. note::
 While placing cells randomly, it will be ensured that they do not occupy excessive volume.
 The ratio of the total cell volume to the partition volume, known as the packing factor,
 should not exceed 0.4.

ParallelArrayPlacement
**********************

*Class*: :class:`bsb.placement.ParallelArrayPlacement
<.placement.arrays.ParallelArrayPlacement>`

This class place cells in an aligned array, it create a lattice with fixed spacing and with the desired angle.
It is necessary to specify ``spacing_x`` and ``angle`` attributes.

.. tab-set-code::

    .. code-block:: json

        "cell_types": {
            "my_cell": {
                "spatial": {
                    "count": 100,
                    "radius": 1
                }
            }
        },

        "placement": {
            "place_on_flat_array":{
                "strategy": "bsb.placement.particle.ParallelArrayPlacement",
                "partitions": "my_layer",
                "cell_types": ["my_cell"],
                "spacing_x": 10,
                "angle": 0
            }
        },

    .. code-block:: python

      config.cell_types.add(
        "my_cell",
        spatial=dict(radius=1, count=100)
      )
      config.placement.add(
        "place_on_flat_array",
        strategy="bsb.placement.ParallelArrayPlacement",
        partitions=["my_layer"],
        cell_types=["my_cell"],
        spacing_x=10,
        angle=0
      )




FixedPositions
**************

*Class*: :class:`bsb.placement.FixedPositions <.placement.strategy.FixedPositions>`

This class places the cells in fixed positions specified in the attribute ``positions``.

* ``positions``: a list of 3D points where the neurons should be placed. For example:

.. tab-set-code::

    .. code-block:: json

        "cell_types": {
            "my_cell": {
                "spatial": {
                    "count": 2,
                    "radius": 2
                }
            }
        },

        "placement": {
            "place_in_fixed_position":{
                "strategy": "bsb.placement.particle.FixedPositions",
                "partitions": "my_layer",
                "cell_types": ["my_cell"],
                "positions": [[0,0,0],[20,20,20]]
            }
        },

    .. code-block:: python

      config.cell_types.add(
        "my_cell",
        spatial=dict(radius=2, count=2)
      )
      config.placement.add(
        "place_in_fixed_position",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["my_cell"],
        positions=[[0,0,0],[20,20,20]]
      )

In this case, we place two cells of type ``my_cell`` at fixed positions
with coordinates [0, 0, 0] and [20, 20, 20].
