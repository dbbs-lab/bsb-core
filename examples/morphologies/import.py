# Import a morphology from a file
from bsb.morphologies import Morphology

m = Morphology.from_swc("my_file.swc")
print(f"My morphology has {len(m)} points and {len(m.branches)} branches.")

# Store it in a MorphologyRepository to use it later.
from bsb.storage import Storage

store = Storage("hdf5", "morphologies.hdf5")
store.morphologies.save("MyCell", m)
