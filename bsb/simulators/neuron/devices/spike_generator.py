from ..adapter import NeuronDevice
from ....simulation.results import SimulationRecorder, PresetPathMixin, PresetMetaMixin
from ....helpers import listify_input
from ....functions import poisson_train
from ....reporting import report, warn
import numpy as np


class SpikeGenerator(NeuronDevice):
    defaults = {"record": True}

    def implement(self, target, cell, section):
        if not hasattr(section, "available_synapse_types"):
            raise Exception(
                "{} {} targetted by {} has no synapses".format(
                    cell.__class__.__name__, ",".join(section.labels), self.name
                )
            )
        for synapse_type in self.synapses:
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
        if "weight" not in self.parameters:
            self.parameters["weight"] = 1
        self.synapses = listify_input(self.synapses)

    def create_patterns(self):
        report("Creating spike generator patterns for '{}'".format(self.name), level=3)
        patterns = {}
        if hasattr(self, "spike_times"):
            return {target: self.spike_times for target in self.get_targets()}
        interval = float(self.parameters["interval"])
        number = int(self.parameters["number"])
        start = float(self.parameters["start"])
        noise = "noise" in self.parameters and self.parameters["noise"]
        if not noise:
            pattern = [start + i * interval for i in range(number)]
        for target in self.get_targets():
            frequency = 1.0 / interval
            duration = interval * number
            if noise:
                pattern = list(poisson_train(frequency, duration, start))
            patterns[target] = pattern
            if self.record:
                self.adapter.result.add(GeneratorRecorder(self, target, pattern))
            report("Pattern {} for {}.".format(pattern, target), level=4)
        return patterns

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.patterns[target]


class GeneratorRecorder(PresetPathMixin, PresetMetaMixin, SimulationRecorder):
    def __init__(self, device, target, pattern):
        self.pattern = pattern
        self.meta = {"device": device.name, "target": str(target)}
        self.path = ("recorders", "input", device.name, str(target))

    def get_data(self):
        return np.array(self.pattern)
