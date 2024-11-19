import bsb.options
from bsb import Configuration, Scaffold

bsb.options.verbosity = 3
config = Configuration.default(storage={"engine": "hdf5", "root": "my_network.hdf5"})

config.network.x = 100.0
config.network.y = 200.0
config.network.z = 300.0
config.network.chunk_size = [100, 300, 100]

config.partitions.add("stellate_layer", thickness=100)

config.regions.add(
    "brain_region",
    type="stack",
    children=["stellate_layer"],
)

config.morphologies = [
    dict(
        file="StellateCell.swc",
        parser={
            "tags": {
                "16": ["dendrites", "proximal_dendrites"],
                "17": ["dendrites", "distal_dendrites"],
                "18": ["axon", "axon_initial_segment"],
            }
        },
    )
]

config.cell_types.add(
    "stellate_cell",
    spatial=dict(
        radius=4,
        density=5e-6,
        morphologies=["StellateCell"],
    ),
)

config.placement.add(
    "stellate_placement",
    strategy="bsb.placement.RandomPlacement",
    cell_types=["stellate_cell"],
    partitions=["stellate_layer"],
)

config.connectivity.add(
    "stellate_to_stellate",
    strategy="bsb.connectivity.VoxelIntersection",
    presynaptic=dict(cell_types=["stellate_cell"], morphology_labels=["axon"]),
    postsynaptic=dict(cell_types=["stellate_cell"], morphology_labels=["dendites"]),
)

config.simulations.add(
    "neuronsim",
    simulator="neuron",
    resolution=0.025,
    duration=100,
    temperature=32,
    cell_models=dict(
        stellate_cell=dict(model="Stellate.definitionStellate", parameters=[])
    ),
    connection_models=dict(
        stellate_to_stellate=dict(
            synapses=[{"synapse": "GABA", "weight": 0.001, "delay": 1}]
        )
    ),
    devices=dict(
        spike_generator=dict(
            device="spike_generator",
            start=9,
            number=1,
            interval=0,
            noise=0,
            delay=1,
            weight=0.01,
            targetting={"strategy": "by_id", "ids": {"stellate_cell": [0]}},
            locations={"strategy": "branch", "labels": ["dendrites"]},
            synapses=["AMPA", "NMDA"],
        ),
        vrecorder=dict(
            device="voltage_recorder",
            targetting={
                "strategy": "sphere",
                "radius": 100,
                "origin": [50, 100, 150],
                "cell_models": ["stellate_cell"],
            },
        ),
        synapses_rec=dict(
            device="synapse_recorder",
            synapse_types=["AMPA", "NMDA"],
            targetting={
                "strategy": "sphere",
                "radius": 100,
                "origin": [50, 100, 150],
                "cell_models": ["stellate_cell"],
            },
            locations={"strategy": "branch", "labels": ["dendrites"]},
        ),
    ),
)

scaffold = Scaffold(config)
scaffold.compile(clear=True)
