from ..device import NeuronDevice
from ....simulation.results import SimulationRecorder
from bsb.simulation.targetting import LocationTargetting
from bsb import config
import numpy as np
import itertools


class UnknownSynapseRecorder(Exception):
    pass


@config.node
class SynapseRecorder(NeuronDevice, classmap_entry="synapse_recorder"):
    locations = config.attr(type=LocationTargetting)
    synapse_types = config.list()

    def implement(self, result, cells, connections):
        for target in self.targetting.get_targets(cells, connections):
            for location in self.locations.get_locations(target):
                for synapse in location.section.synapses:
                    if (
                        not self.synapse_types
                        or synapse.synapse_name in self.synapse_types
                    ):
                        _record_synaptic_current(result, synapse)


def _record_synaptic_current(self, result, synapse):
    result.record(synapse._pp._ref_i)
