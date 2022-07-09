import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.exceptions import *
from bsb.storage import Storage, Chunk
from bsb.storage import _util
from test_setup import get_config
from bsb.unittest import timeout
import mpi4py.MPI as MPI
import pathlib


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
        with link.get() as f:
            self.assertEqual("Your Highness?", f.read(), "message")
        with link.get(binary=True) as f:
            self.assertEqual("Your Highness?", f.read().decode("utf-8"), "message")
