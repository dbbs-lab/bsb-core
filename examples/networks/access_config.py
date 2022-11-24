# How to access the configuration of a network
from bsb.core import from_storage

network = from_storage("network.hdf5")
print("My network was configured with", network.configuration)
print("My network has", len(network.configuration.cell_types), "cell types")
(
    # But to avoid some needless typing and repetition,
    network.cell_types is network.configuration.cell_types
    and network.placement is network.configuration.placement
    and "so on"
)
