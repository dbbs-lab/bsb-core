import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import JSONConfig
from scaffold.output import HDF5Formatter
import scaffold.helpers
from scaffold.exceptions import ConfigurableClassNotFoundError


class TestIssues(unittest.TestCase):
    def test_215(self):
        """
            Assert that reconfiguring an HDF5 doesn't exist doesn't create a gimpy empty
            HDF5 file that causes a downstream error.
        """
        config = JSONConfig(file="mouse_cerebellum_cortex")
        self.assertRaises(
            FileNotFoundError,
            HDF5Formatter.reconfigure,
            "thisHDF5_doesntexist.hdf",
            config,
        )

    def test_235(self):
        """
            Tests the import of configurable classes without a `.` in the name.
            Expects a ConfigurableClassNotFoundError because the TestClass doesn't
            exist, but it would throw another error if the issue wasn't resolved.
        """
        self.assertRaises(
            ConfigurableClassNotFoundError,
            scaffold.helpers.get_configurable_class,
            "TestClass",
        )
