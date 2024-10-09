import bsb.options
from bsb import Configuration, Scaffold

bsb.options.verbosity = 3
config = Configuration.default(storage={"engine": "hdf5"})

config.partitions.add("base_layer", thickness=100)
config.partitions.add("top_layer", thickness=100)
config.regions.add(
    "brain_region",
    type="stack",
    children=[
        "base_layer",
        "top_layer",
    ],
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
    strategy="bsb.placement.RandomPlacement",
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
