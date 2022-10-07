from bsb.simulation import SimulationBackendPlugin
from .adapter import NeuronAdapter
from .simulation import NeuronSimulation


__plugin__ = SimulationBackendPlugin(Simulation=NeuronSimulation, Adapter=NeuronAdapter)
