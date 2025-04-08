from dataclasses import dataclass as _dc

from .adapter import SimulatorAdapter as _SimAdapter
from .simulation import Simulation as _Sim


@_dc
class SimulationBackendPlugin:
    Adapter: _SimAdapter
    Simulation: _Sim


def get_simulation_adapter(name: str, comm=None):
    """
    Return the adapter corresponding to the given simulator name.

    :param str name: Name of the simulator.
    :param comm: The mpi4py MPI communicator to use. Only nodes in the communicator
      will participate in the simulation. The first node will idle as the main node.
    """
    from ._backends import get_simulation_adapters

    return get_simulation_adapters()[name](comm=comm)


__all__ = ["SimulationBackendPlugin", "get_simulation_adapter"]
