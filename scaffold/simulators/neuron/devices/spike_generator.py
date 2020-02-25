from ..adapter import NeuronDevice
from ....simulation import TargetsSections
from ....helpers import listify_input
from ....functions import poisson_train


class SpikeGenerator(NeuronDevice):
    def implement(self, target, cell, section):
        for synapse_type in self.synapses:
            if synapse_type in section.available_synapse_types:
                synapse = cell.create_synapse(section, synapse_type)
                pattern = self.get_pattern(target, cell, section, synapse_type)
                synapse.stimulate(pattern=pattern, weight=1)

    def validate_specifics(self):
        self.parameters["weight"] = 1
        self.synapses = listify_input(self.synapses)

    def create_patterns(self):
        print("Creating spike generator patterns for '{}'".format(self.name))
        patterns = {}
        for target in self.get_targets():
            frequency = 1.0 / float(self.parameters["interval"])
            duration = float(self.parameters["interval"]) * int(self.parameters["number"])
            start = float(self.parameters["start"])
            pattern = list(poisson_train(frequency, duration, start))
            patterns[target] = pattern
            print("Pattern", pattern, "for", target)
        return patterns

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.patterns[target]
