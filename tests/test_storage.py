import unittest
import os
from bsb.core import Scaffold
from bsb.config import from_json
from bsb.storage import _util
from bsb.unittest import get_config_path
from bsb.services import MPI
import pathlib


class TestUtil(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.unlink(f"__dsgss__dd{MPI.get_rank()}.txt")

    def test_links(self):
        cfg = from_json(get_config_path("test_single"))
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
        junk = f"__dsgss__dd{MPI.get_rank()}.txt"
        with open(junk, "w") as f:
            f.write("Your Highness?")
        link = _util.syslink(junk)
        self.assertTrue(link.exists())
        with link.get() as f:
            self.assertEqual("Your Highness?", f.read(), "message")
        with link.get(binary=True) as f:
            self.assertEqual("Your Highness?", f.read().decode("utf-8"), "message")
