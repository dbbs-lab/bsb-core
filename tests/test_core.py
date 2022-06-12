import unittest, os, sys, numpy as np, h5py, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb import core
from bsb.exceptions import *
from test_setup import get_config


class TestCore(unittest.TestCase):
    def test_from_hdf5(self):
        # Use the `from_hdf5` function to load a network.
        netw = Scaffold()
        netw.compile(clear=True)
        netw2 = core.from_hdf5(netw.storage.root)
        with self.assertRaises(FileNotFoundError):
            core.from_hdf5("ehehehehehehe")

    @unittest.expectedFailure
    def test_from_hdf5_missing(self):
        # Missing OK leads to network without active config, should probably load default?
        #
        netw = core.from_hdf5("ehehehehehehe2", missing_ok=True)

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
