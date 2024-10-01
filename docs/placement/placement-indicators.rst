#####################
Placement Indications
#####################

`Placement indications` are a subpart of the a ``CellType`` configuration (attribute ``spatial`` of :doc:`/cells/intro`)
and is leveraged to specify properties during `placement strategies` (see :doc:`/placement/placement-strategies`).
These indications can be either related to the cell type itself (e.g. its soma radius) or
allow for the estimation of the number of cells to place in their respective partitions.


General cell properties
-----------------------
These attributes are related to the ``CellType``; among them, only ``radius`` is mandatory:

* :guilabel:`radius`: The radius of the sphere used to approximate cell soma volume.
* :guilabel:`geometry`: dict = config.dict(type=types.any_())
* :guilabel:`morphologies`: List of morphologies references.

Cell counts estimation properties
---------------------------------
The following attributes allow you to set the strategy used to estimate the counts of cell to place in each partition.
You can choose either to compute the counts independently or with respect to another placement counts.

.. warning::
    BSB rounds the number of cells to place in each partition's chunk. The rounding is stochastic, therefore you
    might not get the same counts of cells for two similar reconstructions.

Independent counts estimation strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can choose one of the following attributes to estimate the number of cells to place in its partition(s).

* :guilabel:`count`: Set directly the number of cells to place.
* :guilabel:`density`: Use a density of cell per unit of volume (in cell/um^-3). This will be converted to counts
  based on the partition(s) total volume.
* :guilabel:`planar_density`: Use a density of cell along the `xy` plane per unit of area (in cell/um^-2). Here too,
  density is converted to counts thanks to the partition(s) area along the `xy` plane.
* :guilabel:`density_key`: Leverage a density file, defined here with a reference to a `partitions.keys`
  (see `Voxels` section in :doc:`/topology/partitions`). The number of cells is derived from this volumetric density
  file for each of its voxels.

Relative counts estimation strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies derives the counts of cells to place from on another ``CellType`` placement. You would need therefore
to specify the cell name as a reference:

* :guilabel:`relative_to`: Reference to a ``CellType``.

And then, you can choose one of the following attributes:

* :guilabel:`count_ratio`: Compute the number of cells to place from the ratio between the current cell type
  and a reference cell type.
* :guilabel:`density_ratio`: Similar to ``count_ratio`` but use density instead of cell count.

.. note::
    You can mix counts and densities:
    For instance, you can have a cell A which counts is relative to the one of another cell type B, and B being defined
    from a density value.


Full example
~~~~~~~~~~~~

.. tab-set-code::

    .. code-block:: json

        "regions": {
            "my_region": {
                "type": "stack",
                "children": ["my_layer", "my_second_layer"],
            }
        }
        "partitions": {
            "my_layer": {
                "thickness": 100
            },
            "my_second_layer": {
                "thickness": 200
            },
        }
        "cell_types": {
            "A": {
                "spatial": {
                    "count": 10,
                    "radius": 2
                }
            },
            "B": {
                "spatial": {
                    "relative_to": "A",
                    "density_ratio": 1.5,
                    "radius": 3
                }
            }
        },

        "placement": {
            "place_A":{
                "strategy": "bsb.placement.RandomPlacement",
                "partitions": ["my_layer"],
                "cell_types": ["A"],
            },
            "place_B":{
                "strategy": "bsb.placement.RandomPlacement",
                "partitions": ["my_second_layer"],
                "cell_types": ["B"],
            }
        },

    .. code-block:: python

      config.partitions.add("my_layer", type="layer", thickness=100)
      config.partitions.add("my_second_layer", type="layer", thickness=200)
      config.regions.add(
        "my_region",
        type="stack",
        children=["my_layer", "my_second_layer"]
      )

      config.cell_types.add(
        "A",
        spatial=dict(radius=2, count=10)
      )
      config.cell_types.add(
        "B",
        spatial=dict(radius=3, relative_to="A", density_ratio=1.5)
      )

      config.placement.add(
        "place_A",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_layer"],
        cell_types=["A"],
      )
      config.placement.add(
        "place_B",
        strategy="bsb.placement.RandomPlacement",
        partitions=["my_second_layer"],
        cell_types=["B"],
      )

The example configuration above creates two Layer partitions (``my_layer`` and ``my_second_layer``)
and assigns a random position to ``A`` and ``B`` within their respective layer.
Assuming that ``my_layer`` is big enough to contains both cells
(see :doc:`RandomPlacement strategy</placement/placement-strategies>`), this will place 10 ``A`` and 30
``B`` because the volume of ``my_second_layer`` :math:`v_2` is twice the one of ``my_layer`` :math:`v_1`
and so:

.. math::
    \begin{split}
    counts_{B} & = density_{A} \cdot 1.5 \cdot v_2 \\
    & = \dfrac{10}{v_1} \cdot 1.5 \cdot 2 \cdot v_1 \\
    & = 10 \cdot 1.5 \cdot 2 \\
    & = 30
    \end{split}
