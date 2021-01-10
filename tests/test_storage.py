import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.exceptions import *
from bsb.storage import Storage
from test_setup import get_config


class _ScaffoldDummy:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_cell_types(self):
        return list(self.cfg.cell_types.values())


class TestStorage(unittest.TestCase):
    pass


class TestHDF5Storage(unittest.TestCase):
    def setUp(self):
        self._open_storages = []

    def tearDown(self):
        for s in self._open_storages:
            os.remove(s)
            os.remove(s + ".lck")

    def random_storage(self):
        rstr = "".join(random.choices(string.ascii_uppercase + string.digits, k=15))
        self._open_storages.append(rstr)
        return Storage("hdf5", rstr)

    def test_init(self):
        # Use the init function to instantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        cfg = from_json(get_config("test_single"))
        s = self.random_storage()
        s.create()
        s.init(_ScaffoldDummy(cfg))
        # Test that `init` created the placement sets for each cell type
        for cell_type in cfg.cell_types.values():
            with self.subTest(type=cell_type.name):
                ps = s._PlacementSet(s._engine, cell_type)
                # Test that the placement set is functional after init call
                ps.append_data(np.array([0, 0, 0]), [0])

    def test_renew(self):
        # Use the renew mechanism to reinstantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        cfg = from_json(get_config("test_single"))
        s = self.random_storage()
        s.create()
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        ps.append_data(np.array([0, 0, 0]), [0])
        id = ps.load_identifiers()
        self.assertEqual(
            1,
            len(ps.load_identifiers()),
            "Failure to setup `storage.renew()` test due to chunk reading error.",
        )
        # Spoof a scaffold here, `renew` only requires an object with a
        # `.get_cell_types()` method for its `storage.init` call.
        s.renew(_ScaffoldDummy(cfg))
        self.assertEqual(
            0,
            len(ps.load_identifiers()),
            "`storage.renew()` did not clear placement data.",
        )
