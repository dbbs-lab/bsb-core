from bsb import config
from bsb.simulation.simulation import Simulation
from .cell import NeuronCell
from .connection import NeuronConnection
from .device import NeuronDevice


@config.node
class NeuronSimulation(Simulation):
    """
    Interface between the scaffold model and the NEURON simulator.
    """

    resolution = config.attr(type=float, default=1.0)
    initial = config.attr(type=float, default=-65.0)
    temperature = config.attr(type=float, required=True)

    cell_models = config.dict(type=NeuronCell, required=True)
    connection_models = config.dict(type=NeuronConnection, required=True)
    devices = config.dict(type=NeuronDevice, required=True)
