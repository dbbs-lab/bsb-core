from bsb.simulation import SimulationBackendPlugin
from .adapter import NestAdapter
from .simulation import NestSimulation
from . import devices


__plugin__ = SimulationBackendPlugin(Simulation=NestSimulation, Adapter=NestAdapter)
