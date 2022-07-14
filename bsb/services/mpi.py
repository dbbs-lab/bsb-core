from ._provider import MockProvider
from ..exceptions import DependencyError
import functools
import os
import warnings


class MockCommunicator:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        pass

    def Abort(self):
        print("MPI Abort called on MockCommunicator", flush=True)
        exit(1)

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def allgather(self, obj):
        return [obj]


class MPIProvider(MockProvider):
    @property
    @functools.cache
    def COMM_WORLD(self):
        if any("mpi" in key.lower() for key in os.environ):
            raise DependencyError(
                "MPI execution detected without MPI dependencies."
                + " Please install with `pip install bsb[mpi]` to use MPI."
            )
        return MockCommunicator()
