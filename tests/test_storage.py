import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.exceptions import *
from bsb.storage import Storage, Chunk
from bsb.storage import _util
from test_setup import get_config, timeout
import mpi4py.MPI as MPI
import pathlib


class _ScaffoldDummy:
    def __init__(self, cfg):
        self.cfg = self.configuration = cfg

    def get_cell_types(self):
        return list(self.cfg.cell_types.values())


class TestStorage(unittest.TestCase):
    pass


# WATCH OUT! These tests are super sensitive to race conditions! Especially through use
# of the @on_master etc decorators in storage.py functions under MPI! We need a more
# detailed MPI checkpointing system, instead of a Barrier system. Consecutive barriers
# can cause slippage, where 1 node skips a Barrier, and it causes sync and race issues,
# and eventually deadlock when it doesn't join the others for the last collective Barrier.
class TestHDF5Storage(unittest.TestCase):
    _open_storages = []

    @timeout(10, abort=True)
    def setUp(self):
        MPI.COMM_WORLD.Barrier()

    @timeout(10, abort=True)
    def tearDown(self):
        MPI.COMM_WORLD.Barrier()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not MPI.COMM_WORLD.Get_rank():
            for s in cls._open_storages:
                os.remove(s)

    def random_storage(self):
        rstr = f"random_storage_{len(self.__class__._open_storages)}.hdf5"
        self.__class__._open_storages.append(rstr)
        s = Storage("hdf5", rstr)
        return s

    @timeout(10)
    def test_init(self):
        # Use the init function to instantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        cfg = from_json(get_config("test_single"))
        s = self.random_storage()
        s.create()
        self.assertTrue(os.path.exists(s._root))
        self.assertTrue(s.exists())
        s.init(_ScaffoldDummy(cfg))
        # Test that `init` created the placement sets for each cell type
        for cell_type in cfg.cell_types.values():
            with self.subTest(type=cell_type.name):
                ps = s._PlacementSet(s._engine, cell_type)
                # Test that the placement set is functional after init call
                ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])

    @timeout(10)
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
            ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])
        self.assertEqual(
            1,
            len(ps.load_positions()),
            "Failure to setup `storage.renew()` test due to chunk reading error.",
        )
        MPI.COMM_WORLD.Barrier()
        # Spoof a scaffold here, `renew` only requires an object with a
        # `.get_cell_types()` method for its `storage.init` call.
        s.renew(_ScaffoldDummy(cfg))
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        self.assertEqual(
            0,
            len(ps.load_positions()),
            "`storage.renew()` did not clear placement data.",
        )

    @timeout(10)
    def test_move(self):
        s = self.random_storage()
        old_root = s._root
        s.create()
        self.assertTrue(os.path.exists(s._root))
        s.move(f"2x2{s._root}")
        self.assertFalse(os.path.exists(old_root))
        self.assertTrue(os.path.exists(s._root))
        s.move(old_root)
        self.assertTrue(os.path.exists(old_root))
        self.assertTrue(os.path.exists(s._root))
        self.assertTrue(s.exists())

    @timeout(10)
    def test_remove_create(self):
        s = self.random_storage()
        s.remove()
        self.assertFalse(os.path.exists(s._root))
        self.assertFalse(s.exists())
        s.create()
        self.assertTrue(os.path.exists(s._root))
        self.assertTrue(s.exists())

    def test_eq(self):
        s = self.random_storage()
        s2 = self.random_storage()
        self.assertEqual(s, s, "Same storage should be equal")
        self.assertNotEqual(s, s2, "Diff storages should be unequal")
        self.assertEqual(s.files, s.files, "Singletons equal")
        self.assertNotEqual(s.files, s2.files, "Diff singletons unequal")
        self.assertNotEqual(s.files, s.morphologies, "Diff singletons unequal")
        self.assertEqual(s.morphologies, s.morphologies, "Singletons equal")
        self.assertNotEqual(s.morphologies, s2.morphologies, "Dff singletons unequal")
        self.assertNotEqual(s.morphologies, "hello", "weird comp should be unequal")


class TestUtil(unittest.TestCase):
    def test_links(self):
        cfg = from_json(get_config("test_single"))
        netw = Scaffold(cfg)
        # Needs more tests, these tests basically only test instantiation
        link = _util.link(netw.files, pathlib.Path.cwd(), "sys", "hellooo", "always")
        link = _util.link(netw.files, pathlib.Path.cwd(), "cache", "hellooo", "always")
        link = _util.link(netw.files, pathlib.Path.cwd(), "store", "hellooo", "always")
        with self.assertRaises(ValueError):
            _util.link("invalid", "oh", "invalid", "ah", "ah")
        link = _util.syslink("hellooooo")
        self.assertFalse(link.exists())
        link = _util.storelink(netw.files, "iddd")
        self.assertFalse(link.exists())
        link = _util.nolink()
        self.assertFalse(link.exists())
        junk = f"__dsgss__dd{MPI.COMM_WORLD.Get_rank()}.txt"
        with open(junk, "w") as f:
            f.write("Your Highness?")
        link = _util.syslink(junk)
        self.assertTrue(link.exists())
        f = link.get()
        print(f, f.name, f.read())
        f.close()
        with link.get() as f:
            self.assertEqual("Your Highness?", f.read(), "message")
        with link.get(binary=True) as f:
            self.assertEqual("Your Highness?", f.read().decode("utf-8"), "message")
