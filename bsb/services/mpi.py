import functools
import os

from ..exceptions import DependencyError
from ._util import MockModule


class MPIService:
    def __init__(self):
        self._mpi = MPIModule("mpi4py.MPI")
        self._comm = self._mpi.COMM_WORLD

    def get_communicator(self):
        return self._comm

    def get_rank(self):
        if self._comm:
            return self._comm.Get_rank()
        return 0

    def get_size(self):
        if self._comm:
            return self._comm.Get_size()
        return 1

    def barrier(self):
        if self._comm:
            return self._comm.Barrier()
        pass

    def abort(self, errorcode=1):
        if self._comm:
            return self._comm.Abort(errorcode)
        print("MPI Abort called on MockCommunicator", flush=True)
        exit(errorcode)

    def bcast(self, obj, root=0):
        if self._comm:
            return self._comm.bcast(obj, root=root)
        return obj

    def gather(self, obj, root=0):
        if self._comm:
            return self._comm.gather(obj, root=root)
        return [obj]

    def allgather(self, obj):
        if self._comm:
            return self._comm.allgather(obj)
        return [obj]


class MPIModule(MockModule):
    """
    Module provider of the MPI interface.
    """

    @property
    @functools.cache
    def COMM_WORLD(self):
        if (
            any("mpi" in key.lower() for key in os.environ)
            and "BSB_IGNORE_PARALLEL" not in os.environ
        ):
            raise DependencyError(
                "MPI runtime detected without parallel support."
                + " Please install with `pip install bsb[parallel]`."
                + " Set `BSB_IGNORE_PARALLEL` to ignore this error."
            )
        return None
