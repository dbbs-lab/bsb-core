=========
Placement
=========
This block in the configuration is responsible for placing cells into partitions.
All placement strategies derive from the :class:`~.placement.strategy.PlacementStrategy` class,
and should provide functions to define the positions of each ``CellType`` within a ``Partition`` volume.

BSB offers several built-in strategies (here is a :doc:`list </placement/placement-strategies>`),
or you can implement your own.
The placement data is stored in :doc:`PlacementSets </placement/placement-set>` for each cell type.

Add a placement strategy
========================

In the ``placement`` block, all placement strategies are defined. For each strategy,
it is necessary to specify the references to their related :guilabel:`partitions` and :guilabel:`cell_types`
with the corresponding attributes.

.. tab-set-code::

    .. code-block:: json


        "placement": {
            "place_A_in_my_layer": {
                "strategy": "bsb.placement.RandomPlacement",
                "partitions": [
                    "my_layer"
                ],
                "cell_types": [
                    "A_type"
                ]
            }
        }

    .. code-block:: python

      config.placement.add(
        "place_A_in_my_layer",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["A_type"],
      )


Use indications
===============

When a cell type is created, it is possible to define spatial attributes called
:doc:`placement indications</placement/placement-indicators>`.
These attributes are used by the placement strategy to determine the distribution of cells within the volume.

.. tab-set-code::

    .. code-block:: json

        "cell_types": {
            "A_type": {
                "spatial": {
                    "density": 0.005,
                    "radius": 2.5
                }
            },
            "B_type": {
                "spatial": {
                    "count": 50,
                    "radius": 5
                }
            }
        }

        "placement": {
            "place_A_and_B_in_my_layer": {
                "strategy": "bsb.placement.RandomPlacement",
                "partitions": [
                    "my_layer"
                ],
                "cell_types": [
                    "A_type","B_type"
                ]
            }
        }

    .. code-block:: python

      config.cell_types.add(
        "A_type",
        spatial=dict(radius=2.5, density=0.005)
      )
      config.cell_types.add(
        "B_type",
        spatial=dict(radius=5, count=50)
      )

      config.placement.add(
        "place_A_and_B_in_my_layer",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["A_type","B_type"],
      )

In this example, type A cells are placed with a density of 0.005 cells/µm^3,
while we place 50 type B cells with a radius of 5 µm.


Add dependencies to Placement Strategies
========================================

It may be necessary to place a set of cells only after specific strategies have been executed.
In such cases, you can define a list of strategies as dependencies.
For example, you can create a :guilabel:`secondary_placement` that is executed only after the
:guilabel:`place_A_and_B_in_my_layer` placement has been completed.


.. tab-set-code::

    .. code-block:: json


        "placement": {
            "secondary_placement": {
                "strategy": "bsb.placement.RandomPlacement",
                "partitions": [
                    "my_layer"
                ],
                "cell_types": [
                    "C_type"
                ],
                "depends_on": ["place_A_and_B_in_my_layer"]
            }
        }

    .. code-block:: python

      config.placement.add(
        "secondary_placement",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["C_type"],
        depends_on=["place_A_and_B_in_my_layer"],
      )
