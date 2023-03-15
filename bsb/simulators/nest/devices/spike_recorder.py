from ..device import NestDevice
from .... import config


@config.node
class SpikeGenerator(NestDevice):
    def implement(self, adapter, populations, connections):
        import nest

        targets = self.targetting.get_targets(adapter, populations, connections)
        gen = nest.Create("poisson_generator", params={"rate": 5})
        nest.Connect(gen, targets)
