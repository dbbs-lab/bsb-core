from bsb import config
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import Targetting


class TargetLocation:
    def __init__(self, cell, section, connection=None):
        self.cell = cell
        self.section = section
        self.connection = connection

    def get_synapses(self):
        return self.connection and self.connection.synapses


@config.dynamic(attr_name="device", auto_classmap=True)
class NeuronDevice(DeviceModel):
    targetting = config.attr(type=Targetting, required=True)

    def implement(self, simulation, simdata):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )
