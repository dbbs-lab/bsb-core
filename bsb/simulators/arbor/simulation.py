from bsb import config
from bsb.simulation.simulation import Simulation
from .cell import ArborCell
from .connection import ArborConnection
from .device import ArborDevice


@config.node
class ArborSimulation(Simulation):
    threads = config.attr(type=int)
    profiling = config.attr(type=bool)
    cell_models = config.dict(type=ArborCell, required=True)
    connection_models = config.dict(type=ArborConnection, required=True)
    devices = config.dict(type=ArborDevice, required=True)
