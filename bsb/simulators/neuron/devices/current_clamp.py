from ..device import NeuronDevice
from bsb import config
from bsb.simulation.targetting import LocationTargetting
from ..device import NeuronDevice
import warnings


@config.node
class CurrentClamp(NeuronDevice, classmap_entry="current_clamp"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    amplitude = config.attr(type=float, required=True)
    before = config.attr(type=float, default=None)
    duration = config.attr(type=float, default=None)

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                clamped = False
                for location in self.locations.get_locations(target):
                    if clamped:
                        warnings.warn(f"Multiple current clamps placed on {target}")
                    self._add_clamp(simdata, location)
                    clamped = True

    def _add_clamp(self, simdata, location):
        sx = location.arc(0.5)
        clamp = location.section.iclamp(
            x=sx, delay=self.before, duration=self.duration, amplitude=self.amplitude
        )
        simdata.result.record(clamp._ref_i)
