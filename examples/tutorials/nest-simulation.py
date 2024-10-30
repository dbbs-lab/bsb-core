import bsb.options
from bsb import from_storage

bsb.options.verbosity = 3

scaffold = from_storage("network.hdf5")
config = scaffold.configuration

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
