from bsb_plot import plot_network

import bsb.options
from bsb import Scaffold, fparse_configuration_file

bsb.options.verbosity = 3
config = parse_configuration_file("network_configuration.json", parser="json")

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
    strategy="bsb.placement.RandomPlacement",
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
