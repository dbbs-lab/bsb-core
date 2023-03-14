from bsb import config
from bsb.config import types, compose_nodes
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import CellTargetting
from .connection import NestConnectionSettings


@config.dynamic(attr_name="device", auto_classmap=True, default="custom")
class NestDevice(compose_nodes(NestConnectionSettings, DeviceModel)):
    pass


class ExtNestDevice(NestDevice):
    nest_model = config.attr(type=str, required=True)
    parameters = config.dict(
        type=types.or_(
            types.evaluation(), types.number(), types.distribution(), types.any_()
        )
    )
    targetting = config.attr(type=CellTargetting)
