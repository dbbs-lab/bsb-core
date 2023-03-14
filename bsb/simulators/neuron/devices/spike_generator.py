from .... import config
from ....config import types
from ..device import NeuronDevice
from ....simulation.results import SimulationRecorder
from bsb.simulation.targetting import LocationTargetting
from ....exceptions import *
from ....reporting import report, warn
import numpy as np
from patch import p


@config.node
class SpikeGenerator(NeuronDevice):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    synapses = config.attr(type=types.list())
    parameters = config.catch_all(type=types.any_())

    def implement(self, result, cells, connections):
        for target in self.targetting.get_targets(cells, connections):
            for location in self.locations.get_locations(target):
                for synapse in location.section.synapses:
                    if self.synapses is None or any(
                        syn in str(synapse._pp) for syn in self.synapses
                    ):
                        synapse.stimulate(**self.parameters)
