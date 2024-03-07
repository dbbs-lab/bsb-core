# TODO: check for parallel support in the hdf5 provider, if it has it, provide noop

from ._util import MockModule


class MockedWindowController:
    def __init__(self, comm=None, master=0):
        from . import MPI

        if comm is None:
            comm = MPI
        self._comm = comm
        self._size = comm.get_size()
        self._rank = comm.get_rank()
        self._master = master
        self._mocked = True
        self._closed = False

    @property
    def master(self):
        return self._master

    @property
    def rank(self):
        return self._rank

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self):
        return _NoopLock()

    def write(self):
        return _NoopLock()

    def single_write(self, handle=None, rank=None):
        if rank is None:
            rank = self._master
        fence = Fence(self._rank == rank)
        if self._rank == rank:
            return _NoopLock(handle=handle, fence=fence)
        elif handle:
            return _NoHandle()
        else:
            return fence


class _NoopLock:
    def __init__(self, handle=None, fence=None):
        self._locked = 0
        self._mocked = True
        self._handle = handle
        self._fence = fence

    def locked(self):
        return self._locked > 0

    def __enter__(self):
        self._locked += 1
        return self._acquire_lock()

    def _acquire_lock(self):
        if self._handle is not None:
            return self._handle
        elif self._fence is not None:
            return self._fence

    def __exit__(self, exc_type, exc_value, traceback):
        self._locked -= 1


class Fence:
    def __init__(self, access):
        self._access = access
        self._obj = None
        self._mocked = True

    def guard(self):
        if not self._access:
            raise FencedSignal()

    def share(self, obj):
        self._obj = obj

    def collect(self):
        return self._obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is FencedSignal:
            return True


class _NoHandle:
    def __init__(self):
        self._mocked = True

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FencedSignal(Exception):
    pass


class MPILockModule(MockModule):
    def sync(self, comm=None, master=0):
        return MockedWindowController(comm, master)

    def __call__(self, comm=None, master=None):
        return MockedWindowController(comm, master)
