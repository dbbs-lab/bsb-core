from bsb import config
from bsb.config import types
from bsb.simulation.simulation import Simulation
from .cell import NestCell
from .connection import NestConnection
from .device import NestDevice


@config.node
class NestSimulation(Simulation):
    """
    Interface between the scaffold model and the NEST simulator.
    """

    modules = config.list(type=str)
    threads = config.attr(type=types.int(min=1), default=1)
    resolution = config.attr(type=types.float(min=0.0), default=1.0)
    default_synapse_model = config.attr(type=str, default="static_synapse")
    default_neuron_model = config.attr(type=str, default="iaf_cond_alpha")
    verbosity = config.attr(type=str, default="M_ERROR")

    cell_models = config.dict(type=NestCell, required=True)
    connection_models = config.dict(type=NestConnection, required=True)
    devices = config.dict(type=NestDevice, required=True)

    def boot(self):
        self.is_prepared = False
        self.suffix = ""
        self.multi = False
        self.has_lock = False
        self.global_identifier_map = {}
