from bsb.simulation import SimulationBackendPlugin
from .simulation import ArborSimulation
from .adapter import ArborAdapter
from . import devices


__plugin__ = SimulationBackendPlugin(Simulation=ArborSimulation, Adapter=ArborAdapter)
