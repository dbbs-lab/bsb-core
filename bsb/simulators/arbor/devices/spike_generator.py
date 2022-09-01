from ..adapter import ArborDevice
from ....simulation.results import SimulationRecorder, PresetPathMixin, PresetMetaMixin
from ....exceptions import *
from ....reporting import report, warn
import numpy as np


class SpikeGenerator(ArborDevice):
    defaults = {"record": True}
    casts = {
        "radius": float,
        "origin": [float],
        "synapses": [str],
    }
    required = ["targetting", "device"]

    def implement(self, target):
        # Device implementation is not required in arbor for `spike_generator`s. The
        # recipe integrates the spike events during using `get_pattern(gid)` data while it
        # is evaluating `get_schedule(gid)` for the associated `spike_source_cell`s.
        #
        # `spike_generator`s currently can't target synapses yet! Only spike source cells!
        return []

    def validate_specifics(self):
        import arbor

        for attr, config in vars(self).items():
            schedule_factory = getattr(arbor, attr, None)
            if schedule_factory is not None and attr.endswith("_schedule"):

                def factory():
                    return schedule_factory(**config).events(0, float("inf"))

                self.schedule_factory = factory
                break
        else:
            raise ConfigurationError(
                f"`{self.name}` is missing an arbor schedule definition."
            )

    def create_patterns(self):
        report(f"Creating spike generator patterns for '{self.name}'", level=3)
        targets = self.get_targets()
        patterns = {target: self.schedule_factory() for target in targets}
        if self.record:
            for target, pattern in patterns.items():
                report(f"Recording pattern {pattern} for {target}.", level=4)
                self.adapter.result.add(GeneratorRecorder(self, target, pattern))
        return patterns

    def get_pattern(self, target):
        return self.get_patterns()[target]


class GeneratorRecorder(PresetPathMixin, PresetMetaMixin, SimulationRecorder):
    def __init__(self, device, target, pattern):
        self.pattern = pattern
        self.meta = {"device": device.name, "target": str(target)}
        self.path = ("recorders", "input", device.name, str(target))

    def get_data(self):
        return np.array(self.pattern)
