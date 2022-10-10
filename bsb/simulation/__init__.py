from .adapter import SimulatorAdapter as _SimAdapter
from .simulation import Simulation as _Sim
from dataclasses import dataclass as _dc


@_dc
class SimulationBackendPlugin:
    Adapter: _SimAdapter
    Simulation: _Sim


def get_simulation_adapter(name):
    from ._backends import get_simulation_adapters

    return get_simulation_adapters()[name]
