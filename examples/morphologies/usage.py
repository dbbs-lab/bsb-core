from bsb.core import from_hdf5

# Load the morphology
network = from_hdf5("network.hdf5")
morpho = network.morphologies.load("my_morphology")
print(f"Has {len(morpho)} points and {len(morpho.branches)} branches.")

# Take a branch
special_branch = morpho.branches[3]
# Assign some labels to the whole branch
special_branch.label("axon", "special")
# Assign labels only to the first quarter of the branch
first_quarter = np.arange(len(special_branch)) < len(special_branch) / 4
special_branch.label(first_quarter, "initial_segment")
# Assign random data as the `random_data` property to the branch
special_branch.set_property(random_data=np.random.random(len(special_branch)))
print(f"Random data for each point:", special_branch.random_data)

network.morphologies.save("processed_morphology", morpho)
