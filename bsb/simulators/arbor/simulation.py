from bsb import config
from bsb.simulation.simulation import Simulation
from .cell import ArborCell
from .connection import ArborConnection
from .device import ArborDevice


@config.node
class ArborSimulation(Simulation):
    cell_models = config.slot(type=ArborCell, required=True)
    connection_models = config.slot(type=ArborConnection, required=True)
    devices = config.slot(type=ArborDevice, required=True)
