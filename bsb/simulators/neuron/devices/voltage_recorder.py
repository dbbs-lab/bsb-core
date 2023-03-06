from ..device import NeuronDevice
import numpy as np


class VoltageRecorder(NeuronDevice):
    casts = {"x": float}

    def implement(self, target, location):
        cell = location.cell
        section = location.section
        group = "voltage_recorders"
        if hasattr(self, "group"):
            group = self.group
        if hasattr(self, "x_interval"):
            for x in np.arange(**self.x_interval):
                self.adapter.register_recorder(
                    group, cell, section.record(x), section=section, x=x
                )
        elif hasattr(self, "x"):
            self.adapter.register_recorder(
                group,
                cell,
                section.record(self.x),
                section=section,
                x=self.x,
            )
        else:
            self.adapter.register_recorder(group, cell, section.record(), section=section)

    def validate_specifics(self):
        pass
