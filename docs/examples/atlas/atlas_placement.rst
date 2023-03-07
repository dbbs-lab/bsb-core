Mouse brain atlas based placement
=================================

The BSB supports integration with cell atlases. All that's required is to implement a
:class:`~bsb.topology.partition.Voxels` partition so that the atlas data can be converted
from the atlas raster format, into a framework object. The framework has
:ref:`allen-atlas-integration` out of the box, and this example will use the
:class:`~bsb.topology.partition.AllenStructure`.

After loading shapes from the atlas, we will use a local data file to assign density
values to each voxel, and place cells accordingly.

We start by defining the basics: a region, an ``allen`` partition and a cell type:

.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 12-17,20-26,28-31
  :emphasize-lines: 6

Here, the :guilabel:`mask_source` is not set so BSB will automatically download the 2017 version of
the CCFv3 mouse brain annotation atlas volume from the Allen Institute website.
Use :guilabel:`mask_source` to provide your own nrrd annotation volume.

The :guilabel:`struct_name` refers to the Allen mouse brain region acronym or name.
You can also replace that with :guilabel:`struct_id`, if you're using the numeric identifiers.
You can find the ids, acronyms and names in the Allen Brain Atlas brain region hierarchy file.

If we now place our ``my_cell`` in the ``declive``, it will be placed with a fixed
density of ``0.003/μm^3``:

.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 32-38

If however, we have data of the cell densities available, we can link our ``declive``
partition to it, by loading it as a :guilabel:`source` file:


.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 15-22
  :emphasize-lines: 4-5

The :guilabel:`source` file will be loaded, and the values at the coordinates of the
voxels that make up our partition are associated as a column of data. We use the
:guilabel:`data_keys` to specify a name for the data column, so that in other places we
can refer to it by name.

We need to select which data column we want to use for the density of ``my_cell``, since
we might need to load multiple densities for multiple cell types, or orientations, or
other data. We can do this by specifying a :guilabel:`density_key`:


.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 23-27,29-31
  :emphasize-lines: 5

That's it! If we compile the network, ``my_cell`` will be placed into ``declive`` with
different densities in each voxel, according to the values provided in
``my_cell_density.nrrd``.
