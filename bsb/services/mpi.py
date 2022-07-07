from ._provider import MockProvider
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
            warnings.warn(
                "MPI execution detected without `mpi4py`."
                + " Output useless, errors likely to occur, please install `mpi4py`."
            )
        return MockCommunicator()
