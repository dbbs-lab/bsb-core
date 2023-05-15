from bsb import config
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import Targetting


@config.dynamic(attr_name="device", auto_classmap=True)
class NeuronDevice(DeviceModel):
    targetting = config.attr(type=Targetting, required=True)
