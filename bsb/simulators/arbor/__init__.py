from bsb.simulation import SimulationBackendPlugin
from .simulation import ArborSimulation
from .adapter import ArborAdapter


__plugin__ = SimulationBackendPlugin(Simulation=ArborSimulation, Adapter=ArborAdapter)
