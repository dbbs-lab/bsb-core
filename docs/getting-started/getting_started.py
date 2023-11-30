from bsb.core import Scaffold
from bsb.config import from_json
from bsb.plotting import plot_network
import bsb.options

bsb.options.verbosity = 3
config = from_json("network_configuration.json")

config.partitions.add("top_layer", thickness=100, stack_index=1)
config.regions.add(
    "brain_region",
    type="stack",
    children=[
        "base_layer",
        "top_layer",
    ],
)
config.cell_types.add("top_type", spatial=dict(radius=7, count=10))
config.placement.add(
    "all_placement",
    strategy="bsb.placement.ParticlePlacement",
    cell_types=["base_type", "top_type"],
    partitions=["base_layer"],
)
config.connectivity.add(
    "A_to_B",
    strategy="bsb.connectivity.AllToAll",
    presynaptic=dict(cell_types=["base_type"]),
    postsynaptic=dict(cell_types=["top_type"]),
)

network = Scaffold(config)
network.compile()
plot_network(network)
