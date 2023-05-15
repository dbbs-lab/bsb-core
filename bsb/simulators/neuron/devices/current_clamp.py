from ..device import NeuronDevice
import warnings


class CurrentClamp(NeuronDevice):
    def implement(self, adapter, simulation, simdata):
        result, cells, connections = simdata.results, simdata.cells, simdata.connections
        for model, targets in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for id_ in targets:
                target = simdata.populations[id_]
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
