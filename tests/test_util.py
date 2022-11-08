import os
import sys
import unittest
import numpy as np
import inspect
from bsb.core import Scaffold
from bsb.voxels import VoxelSet
from bsb.exceptions import *
import bsb.unittest
from bsb.unittest import (
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
)


class TestNetworkUtil(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.network = Scaffold(self.cfg, self.storage)
        self.network.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"], morphology_labels=["axon"]),
                postsynaptic=dict(
                    cell_types=["test_cell"], morphology_labels=["dendrites"]
                ),
            ),
        )
        self.network.compile()

    def test_str(self):
        for obj in (
            self.network,
            *self.network.placement.values(),
            *self.network.connectivity.values(),
            self.network.get_placement_set("test_cell"),
            self.network.get_connectivity_set("test_cell_to_test_cell"),
        ):
            self.assertNotEqual(object.__repr__(obj), str(obj))
            self.assertNotEqual(object.__repr__(obj), repr(obj))
