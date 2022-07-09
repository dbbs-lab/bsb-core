"""
Service module. Register or access interfaces that may be provided, mocked or missing, but
should always behave neatly on import.
"""

from ._provider import ErrorProvider as _ErrorProvider
from .mpi import MPIProvider as _MPIProvider
from .mpilock import MPILockProvider as _MPILockProvider

MPI = _MPIProvider("mpi4py.MPI").COMM_WORLD
"""
MPI service
"""
MPILock = _MPILockProvider("mpilock")
"""
MPILock service
"""

from .pool import JobPool as _JobPool


JobPool = _JobPool
"""
JobPool service
"""


def __getattr__(attr):
    return _ErrorProvider(f"{attr} is not a registered service.")


def register_service(attr, provider):
    globals()[attr] = provider


__all__ = ["MPI", "MPILock", "JobPool", "register_service"]
