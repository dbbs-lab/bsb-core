#####################
Placement Indications
#####################

Placement indications are variables used in cell spatial attributes to specify details
for placement strategies, including the number of cells to place, their volume, and their geometry.
Below is a list of accepted placement indications:

* :guilabel:`radius`: The radius of the sphere used to approximate cell volume.
* :guilabel:`density`: The number of cells to place will be determined based on a density value and the partition volume.
* :guilabel:`planar_density`: The number of cells is computed from a planar density value instead from a volume one.
* :guilabel:`count_ratio`: Compute the number of cells to place from the ratio between the current cell type and a reference cell type.
* :guilabel:`density_ratio`: Similar to `count_ratio` but use density instead of cell count.
* :guilabel:`relative_to`: Specify another "CellType" as a reference for `density_ratio` or `count_ratio`.
* :guilabel:`count`: Set the number of cells to place in the selected partition.
* :guilabel:`geometry`: dict = config.dict(type=types.any_())
* :guilabel:`morphologies`: Add morphologies to the cell type.
* :guilabel:`density_key`: Use a nrrd volumetric density file to define spatial distribution of cells.
