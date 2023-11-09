from .... import config
from ....config import types
from ..device import NeuronDevice
from ....simulation.results import SimulationRecorder
from bsb.simulation.targetting import LocationTargetting
from ....exceptions import *
from ....reporting import report, warn
import numpy as np
from patch import p
import random


@config.node
class SpikeGenerator(NeuronDevice, classmap_entry="spike_generator"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    synapses = config.attr(type=types.list())
    parameters = config.catch_all(type=types.any_())

    def implement(self, result, cells, connections):
        n_syn = 0
        for target in self.targetting.get_targets(cells, connections):
            for location in self.locations.get_locations(target):
                sec = location.section
                print(sec)
                print(sec.labels)
                for synapse in sec.synapses:
                    print(synapse.synapse_name)
                    if synapse.synapse_name in self.synapses:
                        print(n_syn)
                        n_syn = n_syn + 1
                        synapse.stimulate(**self.parameters)
