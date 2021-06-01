import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.exceptions import *
from bsb.storage import Storage
from test_setup import get_config, timeout
import mpi4py.MPI as MPI


class _ScaffoldDummy:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_cell_types(self):
        return list(self.cfg.cell_types.values())


class TestStorage(unittest.TestCase):
    pass


class TestHDF5Storage(unittest.TestCase):
    _open_storages = []

    @timeout(3, abort=True)
    def setUp(self):
        MPI.COMM_WORLD.Barrier()

    @timeout(3, abort=True)
    def tearDown(self):
        MPI.COMM_WORLD.Barrier()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for s in cls._open_storages:
            os.remove(s)

    def rstr(self):
        rstr = "".join(random.choices(string.ascii_uppercase + string.digits, k=15))
        rstr = MPI.COMM_WORLD.bcast(rstr, root=0)
        return rstr

    def random_storage(self):
        rstr = None
        rstr = self.rstr()
        if not MPI.COMM_WORLD.Get_rank():
            self.__class__._open_storages.append(rstr)
        s = Storage("hdf5", rstr)
        return s

    @timeout(3)
    def test_init(self):
        # Use the init function to instantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        cfg = from_json(get_config("test_single"))
        s = self.random_storage()
        self.assertTrue(os.path.exists(s._root))
        s.create()
        s.init(_ScaffoldDummy(cfg))
        # Test that `init` created the placement sets for each cell type
        for cell_type in cfg.cell_types.values():
            with self.subTest(type=cell_type.name):
                ps = s._PlacementSet(s._engine, cell_type)
                # Test that the placement set is functional after init call
                ps.append_data(np.array([0, 0, 0]), [0])

    @timeout(3)
    def test_renew(self):
        # Use the renew mechanism to reinstantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        cfg = from_json(get_config("test_single"))
        s = self.random_storage()
        s.create()
        self.assertTrue(os.path.exists(s._root))
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        with ps._engine._master_write() as fence:
            fence.guard()
            ps.append_data(np.array([0, 0, 0]), [0])
        id = ps.load_identifiers()
        self.assertEqual(
            1,
            len(ps.load_identifiers()),
            "Failure to setup `storage.renew()` test due to chunk reading error.",
        )
        MPI.COMM_WORLD.Barrier()
        # Spoof a scaffold here, `renew` only requires an object with a
        # `.get_cell_types()` method for its `storage.init` call.
        s.renew(_ScaffoldDummy(cfg))
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        self.assertEqual(
            0,
            len(ps.load_identifiers()),
            "`storage.renew()` did not clear placement data.",
        )

    @timeout(6)
    def test_move(self):
        s = self.random_storage()
        old_root = s._root
        self.assertTrue(os.path.exists(s._root))
        s.move(self.rstr())
        self.assertFalse(os.path.exists(old_root))
        self.assertTrue(os.path.exists(s._root))
        self.assertTrue(s.exists())
        s.move(old_root)
        self.assertTrue(os.path.exists(old_root))
        self.assertTrue(os.path.exists(s._root))
        MPI.COMM_WORLD.Barrier()

    @timeout(3)
    def test_remove_create(self):
        s = self.random_storage()
        s.remove()
        self.assertFalse(os.path.exists(s._root))
        self.assertFalse(s.exists())
        s.create()
        self.assertTrue(os.path.exists(s._root))
        self.assertTrue(s.exists())
