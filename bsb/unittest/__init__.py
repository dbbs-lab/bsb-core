from .parallel import *
from ..core import Scaffold as _Scaffold
from ..storage import (
    Storage as _Storage,
    get_engine_node as _get_engine_node,
    Chunk as _Chunk,
)
from ..config import Configuration as _Configuration
from ..exceptions import FixtureError
import numpy as _np
import glob as _glob
import os as _os


class NetworkFixture:
    def setUp(self):
        super().setUp()
        kwargs = {}
        if hasattr(self, "cfg"):
            kwargs["config"] = self.cfg
        if hasattr(self, "storage"):
            kwargs["storage"] = self.storage
        self.network = _Scaffold(**kwargs)


class RandomStorageFixture:
    def __init_subclass__(cls, root_factory=None, debug=False, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory
        cls._open_storages = []
        cls._debug_storage = debug

    def setUp(self):
        super().setUp()
        self.storage = self.random_storage()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls._debug_storage:
            for s in cls._open_storages:
                s.remove()

    @classmethod
    def random_storage(cls):
        if cls._rootf is not None:
            rstr = cls._rootf()
        else:
            # Get the engine's storage node default value, assuming it is random
            rstr = _get_engine_node(cls._engine)(engine=cls._engine).root
        s = _Storage(cls._engine, rstr)
        cls._open_storages.append(s)
        return s


class FixedPosConfigFixture:
    def setUp(self):
        super().setUp()
        self.cfg = _Configuration.default(
            cell_types=dict(test_cell=dict(spatial=dict(radius=2, count=100))),
            placement=dict(
                ch4_c25=dict(
                    strategy="bsb.placement.strategy.FixedPositions",
                    partitions=[],
                    cell_types=["test_cell"],
                )
            ),
        )
        self.chunk_size = cs = self.cfg.network.chunk_size
        self.chunks = [
            _Chunk((0, 0, 0), cs),
            _Chunk((0, 0, 1), cs),
            _Chunk((1, 0, 0), cs),
            _Chunk((1, 0, 1), cs),
        ]
        self.cfg.placement.ch4_c25.positions = MPI.bcast(
            _np.vstack(
                (
                    _np.random.random((25, 3)) * cs + [0, 0, 0],
                    _np.random.random((25, 3)) * cs + [0, 0, cs[2]],
                    _np.random.random((25, 3)) * cs + [cs[0], 0, 0],
                    _np.random.random((25, 3)) * cs + [cs[0], 0, cs[2]],
                )
            )
        )


class MorphologiesFixture:
    def __init_subclass__(cls, morpho_filters=None, morpho_suffix="swc", **kwargs):
        super().__init_subclass__(**kwargs)
        cls._morpho_suffix = morpho_suffix
        cls._morpho_filters = morpho_filters

    def setUp(self):
        super().setUp()
        if not hasattr(self, "network"):
            raise FixtureError(
                f"{self.__class__.__name__} uses MorphologiesFixture, which requires a network fixture."
            )
        if MPI.get_rank():
            MPI.barrier()
        else:
            for mpath in get_all_morphology_paths(self._morpho_suffix):
                if self._morpho_filters and all(
                    mpath.find(filter) == -1 for filter in self._morpho_filters
                ):
                    continue
                if mpath.endswith("swc"):
                    self.network.morphologies.import_swc(mpath)
                else:
                    self.network.morphologies.import_file(mpath)
            MPI.barrier()


class NumpyTestCase:
    def assertClose(self, a, b, msg="", /, **kwargs):
        if msg:
            msg += ". "
        return self.assertTrue(
            _np.allclose(a, b, **kwargs), f"{msg}Expected {a}, got {b}"
        )

    def assertAll(self, a, msg="", /, **kwargs):
        trues = _np.sum(a.astype(bool))
        all = _np.product(a.shape)
        if msg:
            msg += ". "
        return self.assertTrue(
            _np.all(a, **kwargs), f"{msg}Only {trues} out of {all} True"
        )

    def assertNan(self, a, msg="", /, **kwargs):
        if msg:
            msg += ". "
        nans = _np.isnan(a)
        all = _np.product(a.shape)
        return self.assertTrue(
            _np.all(a, **kwargs), f"{msg}Only {_np.sum(nans)} out of {all} True"
        )


def get_data_path(*paths):
    return _os.path.abspath(
        _os.path.join(
            _os.path.dirname(__file__),
            "data",
            *paths,
        )
    )


def get_config_path(file):
    return get_data_path(
        "configs", file + (".json" if not file.endswith(".json") else "")
    )


def get_morphology_path(file):
    return get_data_path("morphologies", file)


def get_all_morphology_paths(suffix=""):
    yield from _glob.glob(get_data_path("morphologies", "*" + suffix))
