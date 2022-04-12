==========
Cell Types
==========

A cell types contains information about cell populations. There are 2 categories: cells,
and entities. A cell has a position, while an entity does not. Cells can also have
morphologies and orientations associated with them. On top of that, both cells and
entities support additional arbitrary properties.

A cell type is an abstract description of the population. During placement, the concrete
data is generated in the form of a :class:`~.storage.interfaces.PlacementSet`. These can
then be connected together into :class:`ConnectivitySets
<.storage.interfaces.ConnectivitySet>`. Furthermore, during simulation, cell types are
represented by **cell models**.

.. rubric:: Basic configuration

The :guilabel:`radius` and :guilabel:`density` are the 2 most basic *placement
indicators*, they specify how large and dense the cells in the population generally are.
The :guilabel:`plotting` block allows you to specify formatting details.

.. code-block:: json

  {
    "cell_types": {
      "my_cell_type": {
        "spatial": {
          "radius": 10.0,
          "density": 3e-9
        },
        "plotting": {
          "display_name": "My Cell Type",
          "color": "pink",
          "opacity": 1.0
        }
      }
    }
  }

.. rubric:: Specifying morphologies

If the cell type is represented by morphologies, you can list multiple :class:`selectors
<.placement.indicator.MorphologySelector>` to fetch them from the
:doc:`/morphologies/repository`.

.. code-block:: json

  {
    "cell_types": {
      "my_cell_type": {
        "spatial": {
          "radius": 10.0,
          "density": 3e-9,
          "morphologies": [
            {
              "selector": "by_name",
              "names": ["cells_A_*", "cell_B_2"]
            }
          ]
        },
        "plotting": {
          "display_name": "My Cell Type",
          "color": "pink",
          "opacity": 1.0
        }
      }
    }
  }
