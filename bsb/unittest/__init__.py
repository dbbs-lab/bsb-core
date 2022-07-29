from .parallel import *
from ..storage import Storage, get_engine_node
import numpy as _np
import glob
import os


class RandomStorageFixture:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._open_storages = []

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for s in cls._open_storages:
            s.remove()

    def random_storage(self, root_factory=None, engine="hdf5"):
        if root_factory is not None:
            rstr = root_factory()
        else:
            # Get the engine's storage node default value, assuming it is random
            rstr = get_engine_node(engine)(engine=engine).root
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


def get_data(*paths):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            *paths,
        )
    )


def get_config(file):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "configs",
            file + (".json" if not file.endswith(".json") else ""),
        )
    )


def get_morphology(file):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "morphologies",
            file,
        )
    )


def get_all_morphologies(suffix=""):
    yield from glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "morphologies", "*" + suffix)
        )
    )
