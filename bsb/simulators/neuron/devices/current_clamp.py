from ..device import NeuronDevice


@config.node
class CurrentClamp(NeuronDevice, classmap_entry="iclamp"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    amplitude = config.attr(type=float, required=True)
    before = config.attr(type=float, default=None)
    duration = config.attr(type=float, default=None)

    def implement(self, result, cells, connections):
        for target in self.targetting.get_targets(cells, connections):
            clamped = False
            for location in self.locations.get_locations(target):
                if clamped:
                    warnings.warn(f"Multiple current clamps placed on {target}")
                self._add_clamp(result, location)
                clamped = True

    def _add_clamp(self, results, location):
        sx = location.arc(0.5)
        clamp = location.section.iclamp(
            x=sx, delay=self.before, duration=self.duration, amplitude=self.amplitude
        )
        results.record(clamp._ref_i)
