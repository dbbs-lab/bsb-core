import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import JSONConfig
from scaffold.output import HDF5Formatter


class TestIssues(unittest.TestCase):
    def test_215(self):
        config = JSONConfig(file="mouse_cerebellum")
        self.assertRaises(
            FileNotFoundError,
            HDF5Formatter.reconfigure,
            "thisHDF5_doesntexist.hdf",
            config,
        )
