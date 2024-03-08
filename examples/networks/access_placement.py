# How to access and work with the placement data of a network
import numpy as np

from bsb import from_storage

network = from_storage("network.hdf5")
for cell_type in network.cell_types:
    ps = cell_type.get_placement_set()
    pos = ps.load_positions()
    print(len(pos), cell_type.name, "placed")
    # The positions are an (Nx3) numpy array
    print("The median cell is located at", np.median(pos, axis=0))
