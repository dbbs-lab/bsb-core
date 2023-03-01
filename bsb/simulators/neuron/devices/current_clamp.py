from ..device import NeuronDevice



@config.node
class CurrentClamp(NeuronDevice, classmap_entry="vclamp"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    current = config.attr(
        type=types.or_(float, types.list(type=float, size=3)), required=True
    )
    before = config.attr(type=float, default=None)
    duration = config.attr(type=float, default=None)
    after = config.attr(type=float, default=None)
    holding = config.attr(type=float, default=None)

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
            x = xs,
            delay=self.before, 
            duration=self.duration+self.after,
            amplitude=self.holding
        )
        results.record(clamp._ref_i)
"""

class CurrentClamp(NeuronDevice):
    def implement(self, target, location):
        pattern = self.get_pattern(target, location)
        location.section.iclamp(**pattern)

    def validate_specifics(self):
        pass

    def create_patterns(self):
        return self.parameters

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.get_patterns()
"""