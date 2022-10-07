from bsb import config
from bsb.config import types
from bsb.simulation.connection import ConnectionModel


_str_list = types.list(type=str)


@config.node
class NeuronConnection(ConnectionModel):
    synapse = config.attr(
        type=types.or_(types.dict(type=_str_list), _str_list), required=True
    )
    source = config.attr(type=str, default=None)

    def resolve_synapses(self):
        return self.synapses
