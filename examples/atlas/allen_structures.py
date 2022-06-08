from bsb.voxels import AllenStructureLoader
import numpy as np
import nrrd

# For this example, we'll be looking into the declive:
struct = "DEC"
# Retrieve the raw Allen information on a structure:
node = AllenStructureLoader.find_structure(struct)
print(node)
# Get all of the IDs that are part of this structure:
ids = AllenStructureLoader.get_structure_idset(struct)
# Get the boolean mask of the structure. 1's are part of the structure, 0s aren't.
mask = AllenStructureLoader.get_structure_mask(struct)
# You can use this to mask other images of the brain:
brain_image, _ = nrrd.read("image.nrrd")
struct_image = np.where(mask, brain_image, np.nan)
# Or, if you prefer an array of the values:
struct_values = brain_image[mask]
