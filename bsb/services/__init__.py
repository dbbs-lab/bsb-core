from .mpi import MPIProvider as _MPIProvider
from ._provider import ErrorProvider as _ErrorProvider

MPI = _MPIProvider("mpi4py.MPI").COMM_WORLD


def __getattr__(attr):
    return _ErrorProvider(f"{attr} is not a registered service.")


def register_service(attr, provider):
    globals()[attr] = provider
