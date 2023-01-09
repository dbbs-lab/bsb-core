from bsb import config
from bsb.config import types
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import CellTargetting
from bsb.exceptions import ConfigurationError
from .connection import NestConnectionSettings
import numpy as np


@config.node
class NestDevice(DeviceModel):
    name = config.attr(type=str, key=True)
    device = config.attr(type=str, required=True)
    parameters = config.dict(
        type=types.or_(
            types.evaluation(), types.number(), types.distribution(), types.any_()
        )
    )
    connection = config.attr(type=NestConnectionSettings)
    targetting = config.attr(type=CellTargetting)
    io = config.attr(type=types.in_(["input", "output"]))

    def validate(self):
        if self.io not in ("input", "output"):
            raise ConfigurationError(
                "Attribute io needs to be either 'input' or 'output' in {}".format(
                    self.node_name
                )
            )
        if hasattr(self, "stimulus"):
            stimulus_name = (
                "stimulus"
                if not hasattr(self.stimulus, "parameter_name")
                else self.stimulus.parameter_name
            )
            self.parameters[stimulus_name] = self.stimulus.eval()

    def get_nest_targets(self):
        """
        Return the targets of the stimulation to pass into the nest.Connect call.
        """
        targets = np.array(self.get_targets(), dtype=int)
        return self.adapter.get_nest_ids(targets)
