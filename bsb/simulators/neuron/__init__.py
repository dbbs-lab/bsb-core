from bsb.simulation import SimulationBackendPlugin
from .adapter import NeuronAdapter
from .simulation import NeuronSimulation
from . import devices


__plugin__ = SimulationBackendPlugin(Simulation=NeuronSimulation, Adapter=NeuronAdapter)
