from .adapter import SimulatorAdapter as _SimAdapter
from .simulation import Simulation as _Sim
from dataclasses import dataclass as _dc


@_dc
class SimulationBackendPlugin:
    Adapter: _SimAdapter
    Simulation: _Sim
