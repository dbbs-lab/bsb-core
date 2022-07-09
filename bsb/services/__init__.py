from ._provider import ErrorProvider as _ErrorProvider
from .mpi import MPIProvider as _MPIProvider
from .mpilock import MPILockProvider as _MPILockProvider

MPI = _MPIProvider("mpi4py.MPI").COMM_WORLD
MPILock = _MPILockProvider("mpilock")

from .pool import JobPool


def __getattr__(attr):
    return _ErrorProvider(f"{attr} is not a registered service.")


def register_service(attr, provider):
    globals()[attr] = provider
