import psutil
from bsb import config
from bsb.config import types
from bsb.simulation.simulation import Simulation
from .cell import ArborCell
from .connection import ArborConnection
from .device import ArborDevice


@config.node
class ArborSimulation(Simulation):
    resolution = config.attr(type=types.float(min=0.0), default=0.1)
    profiling = config.attr(type=bool)
    cell_models = config.dict(type=ArborCell, required=True)
    connection_models = config.dict(type=ArborConnection, required=True)
    devices = config.dict(type=ArborDevice, required=True)

    @config.property(default=1)
    def threads(self):
        return self._threads

    @threads.setter
    def threads(self, value):
        self._threads = value if value != "all" else psutil.cpu_count(logical=False)
