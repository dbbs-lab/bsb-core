from ..device import NeuronDevice
from bsb.simulation.targetting import LocationTargetting
from bsb import config


@config.node
class VoltageRecorder(NeuronDevice, classmap_entry="voltage_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})

    def implement(self, adapter, simulation, simdata):
        for target in self.targetting.get_targets(adapter, simulation, simdata):
            for location in self.locations.get_locations(target):
                self._add_voltage_recorder(result, location)

    def _add_voltage_recorder(self, results, location):
        location.section.record()
        results.record(location.section)
