from bsb.simulation import SimulationBackendPlugin
from .adapter import NestAdapter
from .simulation import NestSimulation
from .devices import *


__plugin__ = SimulationBackendPlugin(Simulation=NestSimulation, Adapter=NestAdapter)
