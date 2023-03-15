from ..device import NeuronDevice


class CurrentClamp(NeuronDevice):
    def implement(self, target, location):
        pattern = self.get_pattern(target, location)
        location.section.iclamp(**pattern)

    def implement(self, adapter, result, cells, connections):
        for target in self.targetting.get_targets(adapter, cells, connections):
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
