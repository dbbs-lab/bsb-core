from ..device import NeuronDevice
from bsb.simulation.targetting import LocationTargetting
from bsb import config


@config.node
class VoltageRecorder(NeuronDevice, classmap_entry="voltage_recorder"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                for location in self.locations.get_locations(target):
                    self._add_voltage_recorder(simdata.result, location)

    def _add_voltage_recorder(self, results, location):
        results.record(location.section)
