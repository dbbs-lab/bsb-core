import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import JSONConfig
from scaffold.output import HDF5Formatter
import scaffold.helpers
from scaffold.exceptions import ConfigurableClassNotFoundError


class TestIssues(unittest.TestCase):
    def test_215(self):
        config = JSONConfig(file="mouse_cerebellum")
        self.assertRaises(
            FileNotFoundError,
            HDF5Formatter.reconfigure,
            "thisHDF5_doesntexist.hdf",
            config,
        )

    def test_235(self):
        self.assertRaises(
            ConfigurableClassNotFoundError,
            scaffold.helpers.get_configurable_class,
            "TestClass",
        )
