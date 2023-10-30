from ..device import NeuronDevice
from ....simulation.results import SimulationRecorder
import numpy as np
import itertools


class SynapseRecorder(NeuronDevice):
    defaults = {
        "record_spikes": True,
        "record_current": False,
        "types": None,
    }

    def boot(self):
        pass

    def validate_specifics(self):
        pass

    def implement(self, adapter, simulation, simdata):
        raise NotImplementedError(
            "Ion recorder not re-implemented yet. Open an issue if you need it."
        )
        cell = location.cell
        section = location.section
        recorder_classes = []
        if self.record_spikes:
            recorder_classes.append(SynapticSpikesRecorder)
        if self.record_current:
            recorder_classes.append(SynapticCurrentRecorder)
        for synapse in get_section_synapses(section, self.types):
            for recorder_class in recorder_classes:
                recorder = recorder_class(cell, section, synapse)
                self.adapter.result.add(recorder)


class _SynapticRecorder(SimulationRecorder):
    def __init_subclass__(cls, record=None, slug=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._record = record
        cls._slug = str(slug)

    def __init__(self, cell, section, synapse):
        from patch import p

        point_process = synapse._point_process
        location = str(point_process.get_segment())
        name = str(point_process.__neuron__())
        self.meta = {
            "cell": cell.ref_id,
            "section": cell.sections.index(section),
            "x": point_process.get_segment().x,
            "type": synapse._type,
            "location": location + "." + name,
        }
        p.pop_section()
        self.vectors = self._record(synapse._point_process)

    def get_data(self):
        signal = []
        for v in self.vectors:
            signal.extend(v)
        return np.array(signal)

    def flush(self, block, segment):
        raise NotImplementedError("Flush synapse")


def _record_i(self, point_process):
    from patch import p

    v = p.Vector()
    v.record(point_process._ref_i)
    return [v]


def _record_spikes(self, point_process):
    import patch.objects
    from patch import p

    nc0 = p.NetCon(point_process, None)
    return [
        nc.record()
        for nc in itertools.chain([nc0], point_process._connections.values())
        if isinstance(nc, patch.objects.NetCon)
    ]


class SynapticCurrentRecorder(_SynapticRecorder, slug="current", record=_record_i):
    pass


class SynapticSpikesRecorder(_SynapticRecorder, slug="spikes", record=_record_spikes):
    pass
