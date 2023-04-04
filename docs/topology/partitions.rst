##########
Partitions
##########

.. _voxel-partition:

======
Voxels
======

:class:`Voxel partitions <.topology.partition.Voxels>` are an irregular shape in space,
described by a group of rhomboids, called a :class:`~.voxels.VoxelSet`. Most brain atlases
scan the brain in a 3D grid and publish their data in the same way, usually in the `Nearly
Raw Raster Data format, NRRD <https://pynrrd.readthedocs.io/en/latest/user-guide.html>`_.
In general, whenever you have a voxelized 3D image, a ``Voxels`` partition will help you
define the shapes contained within.

NRRD
----

To load data from NRRD files use the :class:`~.topology.partition.NrrdVoxels`. By
default it will load all the nonzero values in a source file:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "type": "nrrd",
            "source": "data/my_nrrd_data.nrrd",
            "voxel_size": 25
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import NrrdVoxels

        my_voxel_partition = NrrdVoxels(source="data/my_nrrd_data.nrrd", voxel_size=25)

The nonzero values from the ``data/my_nrrd_data.nrrd`` file will be included in the
:class:`~.voxels.VoxelSet`, and their values will be stored on the voxelset as a *data
column*. Data columns can be accessed through the :attr:`~.voxels.VoxelSet.data` property:

.. code-block:: python

    voxels = NrrdVoxels(source="data/my_nrrd_data.nrrd", voxel_size=25)
    vs = voxels.get_voxelset()
    # Prints the information about the VoxelSet, like how many voxels there are etc.
    print(vs)
    # Prints an (Nx1) array with one nonzero value for each voxel.
    print(vs.data)

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
            "type": "nrrd",
            "mask_value": 55,
            "mask_source": "data/brain_structures.nrrd",
            "source": "data/whole_brain_cell_densities.nrrd",
            "voxel_size": 25
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import NrrdVoxels

        partition = NrrdVoxels(
          mask_value=55,
          mask_source="data/brain_structures.nrrd",
          source="data/whole_brain_cell_densities.nrrd",
          voxel_size=25,
        )
        vs = partition.get_voxelset()
        # This prints the density data of all voxels that were tagged with `55`
        # in the mask source file (your brain structure).
        print(vs.data)

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

    .. code-block:: python

        from bsb.topology.partition import NrrdVoxels

        partition = NrrdVoxels(
          mask_value=55,
          mask_source="data/brain_structures.nrrd",
          sources=[
            "data/type1_data.nrrd",
            "data/type2_data.nrrd",
            "data/type3_data.nrrd",
          ],
          voxel_size=25,
        )
        vs = partition.get_voxelset()
        # `data` will be an (Nx3) matrix that contains `type1` in `data[:, 0]`, `type2` in
        # `data[:, 1]` and `type3` in `data[:, 2]`.
        print(vs.data.shape)

.. _data-columns:

.. rubric:: Tagging the data columns with keys

Instead of using the order in which the sources appear, you can add data keys to associate
a name with each column. Data columns can then be indexed as strings:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
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

    .. code-block:: python

        from bsb.topology.partition import NrrdVoxels

        partition = NrrdVoxels(
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
        vs = partition.get_voxelset()
        # Access data columns as strings
        print(vs.data[:, "type1"])
        # Index multiple columns like this:
        print(vs.data[:, "type1", "type3"])

.. _allen-atlas-integration:

Allen Mouse Brain Atlas integration
-----------------------------------

The Allen Institute for Brain Science (``AIBS``) gives free access, through their website, to thousands
of datasets based on experiments on mice and humans.

For the mouse, these datasets are 3D-registered in a Common Coordinate Framework (CCF).
The ``AIBS`` maintains the `Allen Mouse Brain Atlas <https://mouse.brain-map.org/>`_;
a pair of files which defines a mouse brain region ontology, and its spatial segregation
in the ``CCF``:

- The brain region ontology takes the form of a hierarchical tree of brain region, with the root
  (top parent) region defining the borders of the mouse brain and the leafs its finest
  parcellations. It will be later be called ``Allen Mouse Brain Region Hierarchy`` (``AMBRH``)
  Each brain region in the ``AMBRH`` has a unique ``id``, ``name``, and ``acronym``
  which can all be used to refer to the region.
- They also defined a mouse brain ``Annotation volume`` (NRRD file) which provides for each voxel
  of the ``CCF`` the id of the finest region it belongs to according to the brain region ontology.

With the BSB you can be seamlessly integrate any dataset registered in the Allen Mouse Brain CCF
into your workflow using the :class:`~.topology.partition.AllenStructure`.
By default (:guilabel:`mask_volume` is not specified), the
:class:`~.topology.partition.AllenStructure` leverages the 2017 version of the
``CCFv3 Annotation volume``, which it downloads directly from the Allen website. BSB will also
automatically download the ``AMBRH`` that you can use to filter regions, providing any of the
brain region id, name or acronym identifiers.

You can then download any Allen Atlas registered dataset as a local NRRD file, and associate it to
the structure, by specifying it as a source file (through :guilabel:`source`
or :guilabel:`sources`). The ``Annotation volume`` will be converted to a voxel mask,
and the mask will be applied to your source files, thereby selecting the structure from the source
files. Each source file will be converted into a data column on the voxelset:

.. tab-set-code::

    .. code-block:: json

      {
        "partitions": {
          "my_voxel_partition": {
            "type": "allen",
            "struct_name": "VAL",
            "sources": [
              "data/allen_gene_expression_25.nrrd"
            ],
            "keys": ["expression"]
          }
        }
      }

    .. code-block:: python

        from bsb.topology.partition import AllenStructure

        partition = AllenStructure(
          # Loads the "ventroanterolateral thalamic nucleus" from the
          # Allen Mouse Brain Annotation volume
          struct_name="VAL",
          mask_source="data/brain_structures.nrrd",
          sources=[
            "data/allen_gene_expression_25.nrrd",
          ],
          keys=["expression"],
        )
        print("Gene expression values per voxel:", partition.voxelset.expression)
