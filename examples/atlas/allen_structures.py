from bsb.topology import AllenStructure
import numpy as np
import nrrd

# For this example, we'll be looking into the declive:
struct = "DEC"
print("Structure acronym:", struct)
# Retrieve the raw Allen information on a structure:
node = AllenStructure.find_structure(struct)
print("Allen node:", node)
# Get all the IDs that are part of this structure:
ids = AllenStructure.get_structure_idset(struct)
print("Structure IDs:", ids)
# Get the boolean mask of the structure. 1's are part of the structure, 0s aren't.
mask = AllenStructure.get_structure_mask(struct)
print("The structure contains", np.sum(mask), "voxels")
# You can use this to mask other images of the brain, such as a fictitious density file:
brain_image, _ = nrrd.read("densities.nrrd")
struct_image = np.where(mask, brain_image, np.nan)
# Or, if you prefer an array of the values:
struct_values = brain_image[mask]
print("Average density of the structure:", np.mean(struct_values))
