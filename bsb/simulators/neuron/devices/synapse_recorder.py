from ..adapter import NeuronDevice
from ....reporting import report, warn
from arborize import get_section_synapses


class SynapseRecorder(NeuronDevice):
    defaults = {
        "record_spikes": True,
        "record_current": False,
        "type": None,
    }

    def boot(self):
        pass

    def implement(self, target, cell, section):
        recorder_classes = []
        if self.record_spikes:
            recorder_classes.append(SynapticSpikesRecorder)
        if self.record_current:
            recorder_classes.append(SynapticCurrentRecorder)
        for type, synapse in get_section_synapses(section, self.type).items():
            for recorder_class in recorder_classes:
                recorder = recorder_class(cell, section, synapse)
                self.adapter.result.add(recorder)


class SynapticCurrentRecorder(SimulationRecorder):
    pass


class SynapticSpikesRecorder(SimulationRecorder):
    pass
