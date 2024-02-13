# Import a morphology from a file
from bsb.morphologies import parse_morphology_file

morpho = parse_morphology_file("my_file.swc")
print(f"My morphology has {len(morpho)} points and {len(morpho.branches)} branches.")

# Store it in a MorphologyRepository to use it later.
from bsb.storage import Storage

store = Storage("hdf5", "morphologies.hdf5")
store.morphologies.save("MyCell", morpho)
