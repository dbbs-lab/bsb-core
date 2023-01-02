import itertools
from functools import cache

import numpy as np

from bsb import config
from bsb.simulation.connection import ConnectionModel
from bsb.simulation.parameter import Parameter


@config.dynamic(
    attr_name="model_strategy", required=False, default="transceiver", auto_classmap=True
)
class NeuronConnection(ConnectionModel):
    def create_connections(self, simulation, simdata, connections, gids):
        raise NotImplementedError(
            "Cell models should implement the `create_connections` method."
        )


@config.node
class SynapseSpec:
    synapse = config.attr(type=str, required=True)
    parameters = config.list(type=Parameter)

    def __init__(self, synapse_name=None, /, **kwargs):
        if synapse_name is not None:
            self._synapse = synapse_name


@config.node
class TransceiverModel(NeuronConnection, classmap_entry="transceiver"):
    synapses = config.list(
        type=SynapseSpec,
        required=True,
    )
    parameters = config.list(type=Parameter)
    source = config.attr(type=str)

    def create_connections(self, simulation, simdata, cs):
        self.create_transmitters(simdata, cs)
        self.create_receivers(simdata, cs)

    def create_transmitters(self, simdata, cs):
        pre, _ = cs.load_connections().from_(simdata.chunks).as_globals().all()
        pre[:, 0] += simdata.cid_offsets[cs.pre_type]
        locs = np.unique(pre[:, :2], axis=0)
        for loc in locs:
            gid = simdata.transmap[tuple(loc)]
            simdata.cells[loc[0]].insert_transmitter(gid, (loc[1], 0), source=self.source)

    def create_receivers(self, simdata, cs):
        pre, post = cs.load_connections().incoming().to(simdata.chunks).as_globals().all()
        pre[:, 0] += simdata.cid_offsets[cs.pre_type]
        post[:, 0] += simdata.cid_offsets[cs.post_type]
        for pre_loc, post_loc in zip(pre[:, :2], post):
            gid = simdata.transmap[tuple(pre_loc)]
            cell = simdata.cells[post_loc[0]]
            for spec in self.synapses:
                cell.insert_receiver(gid, spec.synapse, post_loc[1:], source=self.source)
