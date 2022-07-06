# Create a network from a custom configuration object.
from bsb.core import Scaffold
from bsb.config import Configuration

cfg = Configuration()
# Let's set a file name for the network
cfg.storage.root = "my_network.hdf5"
# And add a cell type
cfg.cell_types.add(
    "hero_cells",
    spatial=dict(
        radius=2,
        density=1e-3,
    ),
)

# After customizing your configuration, create a network from it.
network = Scaffold(cfg)
network.compile()
