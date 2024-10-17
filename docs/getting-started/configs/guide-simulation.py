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

config.cell_types.add(
    "top_type",
    spatial=dict(
        radius=2,
        density=1e-4,
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

config.simulations.add(
    "basal_activity",
    simulator="nest",
    resolution=0.1,
    duration=5000,
    cell_models={},
    connection_models={},
    devices={},
)
config.simulations["basal_activity"].cell_models = dict(
    base_type={"model": "iaf_cond_alpha"}, top_type={"model": "iaf_cond_alpha"}
)

config.simulations["basal_activity"].connection_models = dict(
    A_to_B=dict(synapse=dict(model="static_synapse", weight=100, delay=1))
)

config.simulations["basal_activity"].devices = dict(
    general_noise=dict(
        device="poisson_generator",
        rate=20,
        targetting={"strategy": "cell_model", "cell_models": ["top_type"]},
        weight=40,
        delay=1,
    ),
    base_layer_record=dict(
        device="spike_recorder",
        delay=0.1,
        targetting={"strategy": "cell_model", "cell_models": ["base_type"]},
    ),
    top_layer_record=dict(
        device="spike_recorder",
        delay=0.1,
        targetting={"strategy": "cell_model", "cell_models": ["top_type"]},
    ),
)

network = Scaffold(config)
network.compile()

network.run_simulation("basal_activity")
