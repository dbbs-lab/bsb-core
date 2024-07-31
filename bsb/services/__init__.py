"""
Service module. Register or access interfaces that may be provided, mocked or missing, but
should always behave neatly on import.
"""

from ._util import ErrorModule as _ErrorModule
from .mpi import MPIService as _MPIService
from .mpilock import MPILockModule as _MPILockModule

MPI = _MPIService()
"""
MPI service
"""
MPILock = _MPILockModule("mpilock")
"""
MPILock service
"""

from .pool import JobPool as _JobPool  # noqa
from .pool import WorkflowError, pool_cache

JobPool = _JobPool
"""
JobPool service
"""


def __getattr__(attr):
    return _ErrorModule(f"{attr} is not a registered service.")


def register_service(attr, provider):
    globals()[attr] = provider


__all__ = ["MPI", "MPILock", "JobPool", "register_service", "WorkflowError", "pool_cache"]
