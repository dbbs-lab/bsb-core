from .... import config
from ....config import types
from ..device import NeuronDevice
from bsb.simulation.targetting import LocationTargetting


@config.node
class SpikeGenerator(NeuronDevice, classmap_entry="spike_generator"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    synapses = config.list()
    parameters = config.catch_all(type=types.any_())

    def implement(self, adapter, simulation, simdata):
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                for location in self.locations.get_locations(target):
                    for synapse in location.section.synapses:
                        if self.synapses is None or any(
                            syn in str(synapse._pp) for syn in self.synapses
                        ):
                            synapse.stimulate(**self.parameters)
