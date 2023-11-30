from bsb_plotting import plot_network

import bsb.options
from bsb.config import from_json
from bsb.core import Scaffold
from bsb.topology import Stack

bsb.options.verbosity = 3
config = from_json("network_configuration.json")

config.partitions.add("top_layer", thickness=100, stack_index=1)
config.regions["brain_region"] = Stack(
    children=[
        "base_layer",
        "top_layer",
    ]
)
config.morphologies = [
    "neuron_A.swc",
    {"name": "neuron_B", "file": "neuron2.swc"},
]

config.cell_types.base_type.spatial.morphologies = ["neuron_A"]

config.morphologies.append(
    {"name": "neuron_NM", "file": "nm://cell005_GroundTruth"},
)

config.cell_types.add(
    "top_type",
    spatial=dict(
        radius=7,
        count=10,
        morphologies=["neuron_B", "neuron_NM"],
    ),
)
config.placement.add(
    "all_placement",
    strategy="bsb.placement.ParticlePlacement",
    cell_types=["base_type", "top_type"],
    partitions=["base_layer"],
)
config.connectivity.add(
    "A_to_B",
    strategy="bsb.connectivity.VoxelIntersection",
    presynaptic=dict(cell_types=["base_type"]),
    postsynaptic=dict(cell_types=["top_type"]),
)

network = Scaffold(config)

network.compile()
plot_network(network)
