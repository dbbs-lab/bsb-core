import bsb.options
from bsb import Configuration, Scaffold

bsb.options.verbosity = 3
config = Configuration.default(storage={"engine": "hdf5", "root": "network.hdf5"})

config.network.x = 200.0
config.network.y = 200.0
config.network.z = 200.0

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
    dict(name="neuron_B", file="neuron2.swc"),
]

config.cell_types.add(
    "base_type",
    spatial=dict(
        radius=2.5,
        density=3.9e-4,
        morphologies=["neuron_A"],
    ),
)


config.cell_types.add(
    "top_type",
    spatial=dict(
        radius=7,
        count=10,
        morphologies=["neuron_B"],
    ),
)

config.placement.add(
    "base_placement",
    strategy="bsb.placement.RandomPlacement",
    cell_types=["base_type"],
    partitions=["base_layer"],
)
config.placement.add(
    "top_placement",
    strategy="bsb.placement.RandomPlacement",
    cell_types=["top_type"],
    partitions=["top_layer"],
)

config.connectivity.add(
    "A_to_B",
    strategy="bsb.connectivity.VoxelIntersection",
    presynaptic=dict(cell_types=["base_type"]),
    postsynaptic=dict(cell_types=["top_type"]),
)

scaffold = Scaffold(config)
scaffold.compile(clear=True)
