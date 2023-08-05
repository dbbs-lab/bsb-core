from ..device import NeuronDevice
from ....simulation.results import SimulationRecorder
import numpy as np
import itertools


class IonRecorder(NeuronDevice):
    defaults = {
        "record_current": True,
        "record_concentration": False,
    }

    required = ["ion"]

    def boot(self):
        pass

    def validate_specifics(self):
        pass

    def implement(self, target, location):
        cell = location.cell
        section = location.section
        recorders = []
        if self.record_concentration:
            icon = IonicConcentrationRecorder(cell, section, self.ion)
            self.adapter.result.add(icon)
        if self.record_current:
            ii = IonicCurrentRecorder(cell, section, self.ion)
            self.adapter.result.add(ii)


class _IonicRecorder(SimulationRecorder):
    def __init_subclass__(cls, record=None, slug=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._record = record
        cls._slug = str(slug)

    def __init__(self, cell, section, ion):
        from patch import p

        self.inputs = (cell, section, ion)
        self.vectors = self._record(ion, section(0.5))

    def get_data(self):
        signal = []
        for v in self.vectors:
            signal.extend(v)
        return np.array(signal)


def _record_i(self, ion, segment):
    from patch import p

    return [p.record(getattr(getattr(segment, f"{ion}_ion"), f"_ref_i{ion}"))]


def _record_c(self, ion, segment):
    from patch import p

    return [p.record(getattr(getattr(segment, f"{ion}_ion"), f"_ref_{ion}i"))]


class IonicCurrentRecorder(_IonicRecorder, slug="current", record=_record_i):
    pass


class IonicConcentrationRecorder(_IonicRecorder, slug="concentration", record=_record_c):
    pass
