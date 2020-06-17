from ..adapter import NeuronDevice
from ....simulation import TargetsSections
from ....helpers import listify_input
from ....functions import poisson_train
from ....reporting import report, warn


class SpikeGenerator(NeuronDevice):
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
        self.parameters["weight"] = 1
        self.synapses = listify_input(self.synapses)

    def create_patterns(self):
        report("Creating spike generator patterns for '{}'".format(self.name), level=3)
        patterns = {}
        for target in self.get_targets():
            frequency = 1.0 / float(self.parameters["interval"])
            duration = float(self.parameters["interval"]) * int(self.parameters["number"])
            start = float(self.parameters["start"])
            pattern = list(poisson_train(frequency, duration, start))
            patterns[target] = pattern
            report("Pattern {} for {}.".format(pattern, target), level=4)
        return patterns

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        return self.patterns[target]
