from bsb import config
from bsb.config import types
from bsb.simulation.connection import ConnectionModel
from bsb.simulation.parameter import Parameter


@config.dynamic(
    attr_name="model_strategy", required=False, default="transceiver", auto_classmap=True
)
class NeuronConnection(ConnectionModel):
    pass


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

    def create_connections(self):
        pass
