##########
Partitions
##########

======
Voxels
======

:class:`Voxel partitions <.topology.partition.Voxels>` are an irregular shape in space,
described by a group of rhomboids, called a :class:`~.voxels.VoxelSet`. The voxel
partition needs to be configured with a :class:`~.voxels.VoxelLoader` to load the voxelset
from somewhere. Most brain atlases scan the brain in a 3D grid and publish their data in
the same way, usually in the Nearly Raw Raster Data format, NRRD. In general, whenever you
have a voxelized 3D image, a ``Voxels`` partition will help you define the shapes
contained within.

NRRD
----

To load data from NRRD files use the :class:`~.voxels.NrrdVoxelLoader`. By
default it will load all the nonzero values in a source file:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "voxels": {
              "type": "nrrd",
              "source": "data/my_nrrd_data.nrrd",
              "voxel_size": 25
            }
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import Voxels
        from bsb.voxels import NrrdVoxelLoader

        loader = NrrdVoxelLoader(source="data/my_nrrd_data.nrrd", voxel_size=25)
        partition = Voxels(voxels=loader)

The loader has a :meth:`~.voxels.VoxelLoader.get_voxelset` method to access the loaded
:class:`~.voxels.VoxelSet`. The nonzero values will be stored on the
:class:`~.voxels.VoxelSet` as a *data column*. Data columns can be accessed through the
:attr:`~.voxels.VoxelSet.data` property:

.. code-block:: python

    loader = NrrdVoxelLoader(source="data/my_nrrd_data.nrrd", voxel_size=25)
    vs = loader.get_voxelset()
    # Prints the information about the VoxelSet, like how many there are etc.
    print(vs)
    # Prints an (Nx1) array with one nonzero value for each selected voxel.
    print(vs.data.shape)
    partition = Voxels(voxels=loader)

.. rubric:: Using masks

Instead of capturing the nonzero values, you can give a :guilabel:`mask_value` to select
all voxels with that value. Additionally, you can specify a dedicated NRRD file that
contains a mask, the :guilabel:`mask_source`, and fetch the data of the source file(s)
based on this mask. This is useful when one file contains the shapes of certain brain
structure, and other files contain cell population density values, gene expression values,
... and you need to fetch the values associated to your brain structure:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "voxels": {
              "type": "nrrd",
              "mask_value": 55,
              "mask_source": "data/brain_structures.nrrd",
              "source": "data/whole_brain_cell_densities.nrrd",
              "voxel_size": 25
            }
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import Voxels
        from bsb.voxels import NrrdVoxelLoader

        loader = NrrdVoxelLoader(
          mask_value=55,
          mask_source="data/brain_structures.nrrd",
          source="data/whole_brain_cell_densities.nrrd",
          voxel_size=25,
        )
        vs = loader.get_voxelset()
        # This prints the density data of all voxels that were tagged with `55`
        # in the mask source file (your brain structure).
        print(vs.data)
        partition = Voxels(voxels=loader)

.. rubric:: Using multiple source files

It's possible to use multiple source files. If no mask source is applied, a supermask will
be created from all the source file selections, and in the end, this supermask is applied
to each source file. Each source file will generate a data column, in the order that they
appear in the :guilabel:`sources` attribute:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "voxels": {
              "type": "nrrd",
              "mask_value": 55,
              "mask_source": "data/brain_structures.nrrd",
              "sources": [
                "data/type1_data.nrrd",
                "data/type2_data.nrrd",
                "data/type3_data.nrrd",
              ],
              "voxel_size": 25
            }
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import Voxels
        from bsb.voxels import NrrdVoxelLoader

        loader = NrrdVoxelLoader(
          mask_value=55,
          mask_source="data/brain_structures.nrrd",
          sources=[
            "data/type1_data.nrrd",
            "data/type2_data.nrrd",
            "data/type3_data.nrrd",
          ],
          voxel_size=25,
        )
        vs = loader.get_voxelset()
        # `data` will be an (Nx3) matrix that contains `type1` in `data[:, 0]`, `type2` in
        # `data[:, 1]` and `type3` in `data[:, 2]`.
        print(vs.data.shape)
        partition = Voxels(voxels=loader)

.. rubric:: Tagging the data columns with keys

Instead of using the order in which the sources appear, you can add data keys to associate
a name with each column. Data columns can then be indexed as strings:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "voxels": {
              "type": "nrrd",
              "mask_value": 55,
              "mask_source": "data/brain_structures.nrrd",
              "sources": [
                "data/type1_data.nrrd",
                "data/type2_data.nrrd",
                "data/type3_data.nrrd",
              ],
              "keys": ["type1", "type2", "type3"],
              "voxel_size": 25
            }
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import Voxels
        from bsb.voxels import NrrdVoxelLoader

        loader = NrrdVoxelLoader(
          mask_value=55,
          mask_source="data/brain_structures.nrrd",
          sources=[
            "data/type1_data.nrrd",
            "data/type2_data.nrrd",
            "data/type3_data.nrrd",
          ],
          keys=["type1", "type2", "type3"],
          voxel_size=25,
        )
        vs = loader.get_voxelset()
        # Access data columns as strings
        print(vs.data[:, "type1"])
        # Index multiple columns like this:
        print(vs.data[:, "type1", "type3"])
        partition = Voxels(voxels=loader)

Allen Atlas integration
-----------------------

The [Allen Brain Atlas](https://mouse.brain-map.org/) provides NRRD files and brain
structure annotations; with the BSB these can be seamlessly integrated into your workflow
using the :class:`~.voxels.AllenStructureLoader`. In Allen-speak, partitions are
``Structures``, each structure has an id, name and acronym. The BSB accepts any of those
identifiers and will load the Allen Atlas data and select the structure for you. You
can then download any Allen Atlas image as a local NRRD file, and associate it to the
structure:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "voxels": {
              "type": "allen",
              "struct_name": "VAL",
              "sources": [
                "data/allen_gene_expression_25.nrrd"
              ],
              "keys": ["expression"],
              "voxel_size": 25
            }
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import Voxels
        from bsb.voxels import AllenStructureLoader

        loader = AllenStructureLoader(
          # Loads the "ventroanterolateral thalamic nucleus" from the
          # ALlen Mouse Brain Atlas
          struct_name="VAL",
          mask_source="data/brain_structures.nrrd",
          sources=[
            "data/allen_gene_expression_25.nrrd",
          ],
          keys=["expression"],
          voxel_size=25,
        )
        partition = Voxels(voxels=loader)
