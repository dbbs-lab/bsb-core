from ..adapter import NeuronDevice
from ....simulation.results import SimulationRecorder, PresetPathMixin, PresetMetaMixin
from ....helpers import listify_input
from ....exceptions import *
from ....functions import poisson_train
from ....reporting import report, warn
import numpy as np


class SpikeGenerator(NeuronDevice):
    defaults = {"record": True}

    def implement(self, target, location):
        cell = location.cell
        section = location.section
        if not hasattr(section, "available_synapse_types"):
            raise Exception(
                "{} {} targetted by {} has no synapses".format(
                    cell.__class__.__name__, ",".join(section.labels), self.name
                )
            )
        for synapse_type in location.get_synapses() or self.synapses:
            if synapse_type in section.available_synapse_types:
                synapse = cell.create_synapse(section, synapse_type)
                pattern = self.get_pattern(target, cell, section, synapse_type)
                synapse.stimulate(pattern=pattern, weight=1)
            else:
                warn(
                    "{} targets {} {} with a {} synapse but it doesn't exist on {}".format(
                        self.name,
                        cell.__class__.__name__,
                        cell.ref_id,
                        synapse_type,
                        ",".join(section.labels),
                    )
                )

    def validate_specifics(self):
        if not hasattr(self, "spike_times") and not hasattr(self, "parameters"):
            raise ConfigurationError(
                f"{self.name} is missing `spike_times` or `parameters`"
            )
        self.synapses = listify_input(self.synapses)

    def create_patterns(self):
        report("Creating spike generator patterns for '{}'".format(self.name), level=3)
        targets = self.get_targets()
        if hasattr(self, "spike_times"):
            pattern = self.spike_times
            if self.record:
                for target in targets:
                    self.adapter.result.add(GeneratorRecorder(self, target, pattern))
            patterns = {target: pattern for target in targets}
        else:
            interval = float(self.parameters["interval"])
            number = int(self.parameters["number"])
            start = float(self.parameters["start"])
            noise = "noise" in self.parameters and self.parameters["noise"]
            frequency = 1.0 / interval
            duration = interval * number
            if not noise:
                # Create only 1 copy of the pattern array, might be surprising
                # for tinkering users, but in the framework the created patterns
                # should be used read only in `get_pattern(target)` to pass as
                # input to a VecStim.
                pattern = [start + i * interval for i in range(number)]
                patterns = {target: pattern for target in targets}
            else:
                patterns = {
                    target: list(poisson_train(frequency, duration, start))
                    for target in targets
                }
            if self.record:
                for target, pattern in patterns.items():
                    self.adapter.result.add(GeneratorRecorder(self, target, pattern))
                    report("Pattern {} for {}.".format(pattern, target), level=4)
        return patterns

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.get_patterns()[target]


class GeneratorRecorder(PresetPathMixin, PresetMetaMixin, SimulationRecorder):
    def __init__(self, device, target, pattern):
        self.pattern = pattern
        self.meta = {"device": device.name, "target": str(target)}
        self.path = ("recorders", "input", device.name, str(target))

    def get_data(self):
        return np.array(self.pattern)
