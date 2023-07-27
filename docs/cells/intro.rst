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

.. rubric:: Specifying spatial density

You can set the spatial distribution for each cell type present in a
:class:`~.topology.partition.NrrdVoxels` partition.

To do so, you should first attach your nrrd volumetric density file(s) to the partition with either
the :guilabel:`source` or :guilabel:`sources` blocks.
Then, label the file(s) with the :guilabel:`keys` list block and refer to the :guilabel:`keys`
in the :guilabel:`cell_types` with :guilabel:`density_key`:

.. code-block:: json

  {
    "partitions": {
      "declive": {
        "type": "nrrd",
        "sources": ["first_cell_type_density.nrrd",
                    "second_cell_type_density.nrrd"],
        "keys": ["first_cell_type_density",
                 "second_cell_type_density"]
        "voxel_size": 25,
      }
    }
    "cell_types": {
      "first_cell_type": {
        "spatial": {
          "radius": 10.0,
          "density_key": "first_cell_type_density"
        },
        "plotting": {
          "display_name": "First Cell Type",
          "color": "pink",
          "opacity": 1.0
        }
      },
      "first_cell_type": {
        "spatial": {
          "radius": 5.0,
          "density_key": "second_cell_type_density"
        },
        "plotting": {
          "display_name": "Second Cell Type",
          "color": "#0000FF",
          "opacity": 0.5
        }
      }
    }
  }

The nrrd files should contain voxel based volumetric density in unit of cells / voxel volume,
where the voxel volume is in cubic unit of :guilabel:`voxel_size`.
i.e., if :guilabel:`voxel_size` is in µm then the density file is in cells/µm^3.

.. rubric:: Specifying morphologies

If the cell type is represented by morphologies, you can list multiple :class:`selectors
<.morphologies.selector.MorphologySelector>` to fetch them from the
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
              "select": "by_name",
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
