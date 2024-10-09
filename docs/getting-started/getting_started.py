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
