Atlas based placement
=====================

The BSB supports integration with cell atlases. All that's required is to implement a
:class:`~bsb.voxels.VoxelLoader` so that the atlas data can be converted from the atlas
raster format, into a framework object. The framework has :ref:`allen-atlas-integration`
out of the box, and this example will use the :class:`~bsb.voxels.AllenStructureLoader` as
voxel loader.

After loading shapes from the atlas, we will use a local data file to assign density
values to each voxel, and place cells accordingly.

We start by defining the basics: a region, a cell type and a partition component, of
:guilabel:`type` ``voxels``, with an ``allen`` typed voxel loader:

.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 12-19,22-29,31-34
  :emphasize-lines: 6,8

The :guilabel:`struct_name` refers to the Allen structure acronym or name. You can also
replace that with :guilabel:`struct_id`, if you're using the numeric identifiers. You can
find the ids, acronyms and names in the Allen Brain Atlas.

If we now place our ``my_cell`` in the ``declive``, it will be placed with a fixed
density of ``0.003/μm^3``:

.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 35-41

If however, we have an image of the cell densities available, we can link our ``declive``
partition to it, by loading it as a :guilabel:`source` file:


.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 15-25
  :emphasize-lines: 6-7

The :guilabel:`source` file will be loaded, and the values at the coordinates of the
voxels that make up our partition are associated as a column of data. We use the
:guilabel:`data_keys` to specify a name for the data column, so that in other places we
can refer to it by name.

Since we might need to load multiple densities for multiple cell types, or orientations,
or other data, we need to select which data column we want to use for the density of
``my_cell``. We can do this by giving a :guilabel:`density_key`:


.. literalinclude:: ../../../examples/atlas/allen_structure.json
  :language: json
  :lines: 26-34
  :emphasize-lines: 5

That's it! If we compile the network, ``my_cell`` will be placed into ``declive`` with
different densities in each voxel, according to the values provided in
``my_cell_density.nrrd``.
