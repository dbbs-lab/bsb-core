import numpy as np

from bsb import from_storage

scaffold = from_storage("network.hdf5")

# Access the Scaffold's configuration
config = scaffold.configuration
print(f"My network was configured with {config}")
# scaffold.cell_types corresponds to scaffold.configuration.cell_types
print(f"My network has {len(scaffold.cell_types)} cell types")

# Load placement information from the storage.
for type_name, cell_type in scaffold.cell_types.items():
    ps = cell_type.get_placement_set()
    pos = ps.load_positions()
    print(f"{len(pos)} {type_name} placed")
    # The positions are a (Nx3) numpy array
    print("The median cell is located at", np.median(pos, axis=0))

# Load the connection information from the storage
# for a specific connection set
cs = scaffold.get_connectivity_set("A_to_B")
for src_locs, dest_locs in cs.load_connections():
    print(f"Cell id: {src_locs[0]} connects to cell {dest_locs[0]}")
