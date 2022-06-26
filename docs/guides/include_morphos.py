from bsb.core import Scaffold
from bsb.config import from_json
from bsb.topology import Stack
from bsb.plotting import plot_network
import bsb.options

bsb.options.verbosity = 3
config = from_json("network_configuration.json")

config.partitions.add("top_layer", thickness=100, stack_index=1)
config.regions["brain_region"] = Stack(
    children=[
        "base_layer",
        "top_layer",
    ]
)
config.cell_types.base_type.spatial.morphologies = [
    dict(
        names=["my_neuron"],
    )
]
config.cell_types.add(
    "top_type",
    spatial=dict(
        radius=7,
        count=10,
        morphologies=[
            dict(
                select="from_neuromorpho",
                names=[
                    "cell005_GroundTruth",
                    "DD13-10-c8-3",
                    "10_666-GM9-He-Ctl-Chow-BNL16A-CA1Finished2e",
                ],
            )
        ],
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
