import unittest
import os
from bsb.core import Scaffold
from bsb.storage.interfaces import PlacementSet
from bsb.config import Configuration
from bsb import core
from bsb.unittest import RandomStorageFixture


class TestCore(unittest.TestCase):
    def test_from_storage(self):
        # Use the `from_storage` function to load a network.
        netw = Scaffold()
        netw.compile(clear=True)
        core.from_storage(netw.storage.root)
        with self.assertRaises(FileNotFoundError):
            core.from_storage("does_not_exist")

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink("does_not_exist2")
        except Exception:
            pass

    def test_set_netw_root_nodes(self):
        netw = Scaffold()
        # anti-pattern, but testing it anyway, resetting the storage configuration leads
        # to undefined behavior. (Only changing the properties of the storage object is
        # supported).
        netw.storage_cfg = {"root": netw.storage.root, "engine": "hdf5"}
        netw.storage_cfg.root

    def test_set_netw_config(self):
        netw = Scaffold()
        netw.configuration = Configuration.default(regions=dict(x=dict(children=[])))
        self.assertEqual(1, len(netw.regions), "setting cfg failed")

    def test_netw_props(self):
        netw = Scaffold()
        self.assertEqual(0, len(netw.morphologies.all()), "just checking morph prop")

    def test_resize(self):
        cfg = Configuration.default()
        cfg.partitions.add("layer", thickness=100)
        cfg.regions.add("region", children=["layer"])
        netw = Scaffold(cfg)
        netw.resize(x=500, y=500, z=500)
        self.assertEqual(500, netw.network.x, "resize didnt update network node")
        self.assertEqual(500, netw.partitions.layer.data.width, "didnt resize layer")

    def test_get_placement_sets(self):
        cfg = Configuration.default(
            cell_types=dict(my_type=dict(spatial=dict(radius=2, density=1)))
        )
        netw = Scaffold(cfg)
        pslist = netw.get_placement_sets()
        self.assertIsInstance(pslist, list, "should get list of PS")
        self.assertEqual(1, len(pslist), "should have one PS per cell type")
        self.assertIsInstance(pslist[0], PlacementSet, "elements should be PS")


class TestProfiling(RandomStorageFixture, unittest.TestCase, engine_name="hdf5"):
    def setUp(self):
        super().setUp()
        self.netw = Scaffold(Configuration.default(), storage=self.storage)

    def test_profiling(self):
        import bsb.options
        import bsb.profiling

        bsb.options.profiling = True
        self.netw.compile()

        self.assertGreater(
            len(bsb.profiling.get_active_session()._meters), 0, "missing meters"
        )
        bsb.options.profiling = False
