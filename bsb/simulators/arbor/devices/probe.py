from ..device import ArborDevice
from ....simulation.results import SimulationRecorder
from ....exceptions import ConfigurationError


class Probe(ArborDevice):
    required = ["targetting", "probe_type"]

    def get_probe_name(self):
        return f"cable_probe_{self.probe_type}"

    def validate_specifics(self):
        import arbor

        if self.get_probe_name() not in vars(arbor):
            raise ConfigurationError(
                f"`{self.probe_type}` is not a valid probe type for `{self.name}`"
            )

    def implement(self, target):
        import arbor

        probe_args = ("where", "mechanism", "ion", "state")
        kwargs = dict((k, getattr(self, k)) for k in probe_args if hasattr(self, k))
        return [getattr(arbor, self.get_probe_name())(**kwargs)]

    def prepare_samples(self, sim):
        super().prepare_samples(sim)
        for probe_id, handle in zip(self._probe_ids, self._handles):
            self.adapter.result.add(ProbeRecorder(self, sim, probe_id, handle))


class ProbeRecorder:
    def __init__(self, device, sim, probe_id, handle):
        self.path = ("recorders", device.name, *probe_id)
        self.meta = device.get_meta()
        self.meta["probe_id"] = probe_id
        self._sim = sim
        self._handle = handle

    def samples(self):
        return self._sim.samples(self._handle)

    def multi_collect(self):
        for i, sample in enumerate(self.samples()):
            yield ProbeRecorderSample(self, i, sample)


class ProbeRecorderSample(SimulationRecorder):
    def __init__(self, parent, i, sample):
        self.path = tuple(list(parent.get_path()) + [i])
        self.data = sample[0]
        self.meta = parent.meta.copy()
        self.meta["location"] = str(sample[1])

    def get_data(self):
        return self.data
