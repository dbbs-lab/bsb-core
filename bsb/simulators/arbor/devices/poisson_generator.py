from bsb import config
from ..connection import Receiver
from ..device import ArborDevice
import arbor


@config.node
class PoissonGenerator(ArborDevice, classmap_entry="poisson_generator"):
    record = config.attr(type=bool, default=True)
    rate = config.attr(type=float, required=True)
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=float, required=True)

    def implement_probes(self, simdata, gid):
        return []

    def implement_generators(self, simdata, gid):
        target = Receiver(self, None, [-1, -1], [-1, -1], 0).on()
        gen = arbor.event_generator(
            target, self.weight, arbor.poisson_schedule(0, self.rate / 1000, gid)
        )
        return [gen]
