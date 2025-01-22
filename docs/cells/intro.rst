==========
Cell Types
==========

A cell type is an abstract description of a cell population. Cell populations are
placed within `Partitions` according to :doc:`placement indications </placement/placement-indicators>`.
You can also attach morphologies and orientations to them.
During placement, the cell positions are generated in the form of a :doc:`PlacementSet </placement/placement-set>`.
These can then be connected together into :class:`ConnectivitySets
<.storage.interfaces.ConnectivitySet>`. Furthermore, during simulation, cell types are
represented by **cell models**.

.. rubric:: Basic configuration

The :guilabel:`radius` and :guilabel:`density` are the 2 most basic :doc:`placement indications </placement/placement-indicators>`:
they specify how large and dense the cells in the population generally are.
The :guilabel:`plotting` block allows you to specify formatting details.

.. tab-set-code::

    .. code-block:: json

        "cell_types": {
            "my_cell": {
                "spatial": {
                    "density": 3e-9,
                    "radius": 10
                }
                "plotting": {
                  "display_name": "My Cell Type",
                  "color": "pink",
                  "opacity": 1.0
                }
            }
        }

    .. code-block:: python

      config.cell_types.add(
        "my_cell",
        spatial=dict(radius=10, density=3e-9)
        plotting=dict(display_name=" My Cell Type", color="pink",opacity="1.0")
      )

.. rubric:: Specifying spatial density

In the previous example, we were setting the number of cells to place within each partition
based on a single density value. Let's imagine now that you want to describe the spatial
distribution of the cell type spatial density for each voxel within your partition.
This can be achieved with the :ref:`NrrdVoxels <voxel-partition>` partition.

To do so, you should first attach your NRRD volumetric density file(s) to the partition with either
the :guilabel:`source` or :guilabel:`sources` blocks.
Then, label the file(s) with the :guilabel:`keys` list block and refer to the :guilabel:`keys`
in the :guilabel:`cell_types` with :guilabel:`density_key`:

.. tab-set-code::

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
          "second_cell_type": {
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

    .. code-block:: python


        config.partitions.add(
            "declive",
            type="nrrd",
            sources= ["first_cell_type_density.nrrd",
                        "second_cell_type_density.nrrd"],
            keys= ["first_cell_type_density",
                 "second_cell_type_density"],
            voxel_size=25,

        )

        config.cell_types.add(
            "first_cell_type",
            spatial=dict(radius=10, density_key="first_cell_type_density")
            plotting=dict(display_name="First Cell Type", color="pink",opacity="1.0")
        )
        config.cell_types.add(
            "second_cell_type",
            spatial=dict(radius=10, density_key="second_cell_type_density")
            plotting=dict(display_name="First Cell Type", color="#0000FF",opacity="0.5")
        )

The NRRD files should contain voxel based volumetric density in unit of cells / voxel volume,
where the voxel volume is in cubic unit of :guilabel:`voxel_size`.
i.e., if :guilabel:`voxel_size` is in µm then the density file is in cells/µm^3.
This implementation corresponds to an atlas-based reconstruction and you can find an example of
a BSB configuration using the Allen Atlas in :doc:`this section </examples/atlas/atlas_placement>` .

.. rubric:: Specifying morphologies

The easiest way to associate a morphology to a cell type is by referencing the name it is stored under.
There are more advanced ways as well, covered in our guide on :ref:`Morphology Selectors <morphology_selector>` .

.. tab-set-code::

    .. code-block:: json


      {
        "cell_types": {
          "my_cell_type": {
            "spatial": {
              "radius": 10.0,
              "density": 3e-9,
              "morphologies": ["cells_A_*", "cell_B_2"]
            },
            "plotting": {
              "display_name": "My Cell Type",
              "color": "pink",
              "opacity": 1.0
            }
          }
        }
      }

    .. code-block:: python

        config.cell_types.add(
            "my_cell_type",
            spatial=dict(radius=10, density=3e-9,morphologies=["cells_A_*", "cell_B_2"])
            plotting=dict(display_name=" My Cell Type", color="pink",opacity="1.0")
        )

In this case we add two different morphologies labels:
:guilabel:`cell_B_2` add the morphology with this name, :guilabel:`cells_A_*` add all the stored morphologies with name starting with ``cells_A_`` prefix.
You can also apply transformation to your cell morphologies as discussed in :ref:`this section<transform>`.
