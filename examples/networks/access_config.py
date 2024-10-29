# How to access the configuration of a network
from bsb import from_storage

scaffold = from_storage("network.hdf5")
print("My network was configured with", scaffold.configuration)
print("My network has", len(scaffold.cell_types), "cell types")
(
    # But to avoid some needless typing and repetition,
    scaffold.cell_types is scaffold.configuration.cell_types
    and scaffold.placement is scaffold.configuration.placement
    and "so on"
)
