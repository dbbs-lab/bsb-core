from .parallel import *
from ..storage import Storage
import numpy as _np

_storagecount = 0


class RandomStorageFixture:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._open_storages = []

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not MPI.Get_rank():
            for s in cls._open_storages:
                s.remove()

    def random_storage(self, root_factory=None, engine="hdf5"):
        global _storagecount
        if root_factory is None:
            rstr = f"random_storage_{_storagecount}.hdf5"
            _storagecount += 1
        else:
            rstr = root_factory()
        s = Storage(engine, rstr)
        self.__class__._open_storages.append(s)
        return s


class NumpyTestCase:
    def assertClose(self, a, b, msg="", /, **kwargs):
        return self.assertTrue(_np.allclose(a, b, **kwargs), f"Expected {a}, got {b}")

    def assertAll(self, a, msg="", /, **kwargs):
        trues = _np.sum(a.astype(bool))
        all = _np.product(a.shape)
        return self.assertTrue(
            _np.all(a, **kwargs), f"{msg}. Only {trues} out of {all} True"
        )

    def assertNan(self, a, msg="", /, **kwargs):
        nans = _np.isnan(a)
        all = _np.product(a.shape)
        return self.assertTrue(
            _np.all(a, **kwargs), f"{msg}. Only {_np.sum(nans)} out of {all} True"
        )
