import abc

from bsb import config
from bsb.config import types
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import Targetting

try:
    import arbor

    _has_arbor = True
except ImportError:
    _has_arbor = False
    import types as _t

    # Mock missing requirements, as arbor is, like
    # all simulators, an optional dep. of the BSB.
    arbor = _t.ModuleType("arbor")
    arbor.recipe = type("mock_recipe", (), dict())

    def get(*arg):
        raise ImportError("Arbor not installed.")

    arbor.__getattr__ = get


@config.dynamic(attr_name="device", auto_classmap=True, classmap_entry=None)
class ArborDevice(DeviceModel):
    targetting = config.attr(type=Targetting, required=True)
    resolution = config.attr(type=float)
    sampling_policy = config.attr(type=types.in_(["exact"]))

    def __init__(self, **kwargs):
        self._probe_ids = []

    def __boot__(self):
        self.resolution = self.resolution or self.simulation.resolution

    def register_probe_id(self, gid, tag):
        self._probe_ids.append((gid, tag))

    def prepare_samples(self, simdata):
        self._handles = [
            self.sample(simdata.arbor_sim, probe_id) for probe_id in self._probe_ids
        ]

    def sample(self, sim, probe_id):
        schedule = arbor.regular_schedule(self.resolution)
        sampling_policy = getattr(arbor.sampling_policy, self.sampling_policy)
        return sim.sample(probe_id, schedule, sampling_policy)

    def get_samples(self, sim):
        return [sim.samples(handle) for handle in self._handles]

    def get_meta(self):
        attrs = ("name", "sampling_policy", "resolution")
        return dict(zip(attrs, (getattr(self, attr) for attr in attrs)))

    @abc.abstractmethod
    def implement_probes(self, simdata, target):
        pass

    @abc.abstractmethod
    def implement_generators(self, simdata, target):
        pass
