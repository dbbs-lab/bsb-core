from bsb.core import from_hdf5
import numpy as np

# Load the morphology
network = from_hdf5("network.hdf5")
morpho = network.morphologies.load("my_morphology")

# Filter branches
big_branches = [b for b in morpho.branches if np.any(b.radii > 2)]
for b in big_branches:
    # Label all points on the branch as a `big_branch` point
    b.label("big_branch")
    if b.is_terminal:
        # Label the last point on terminal branches as a `tip`
        b.label([-1], "tip")

network.morphologies.save("labelled_morphology", morpho)
