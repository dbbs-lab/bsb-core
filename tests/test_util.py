import os
import sys
import unittest
import numpy as np
import inspect

from scipy.spatial.transform import Rotation

from bsb._util import rotation_matrix_from_vectors
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
            self.network.get_connectivity_set("all_to_all"),
        ):
            self.assertNotEqual(object.__repr__(obj), str(obj))
            self.assertNotEqual(object.__repr__(obj), repr(obj))


class TestRotationUtils(unittest.TestCase):
    def test_rotation_matrix_from_vectors(self):
        vec1 = [0, 0, 1]
        vec2 = [0, 1, 0]
        err1 = [0, 0, 0]
        err2 = [np.nan, 0, 1]
        self.assertTrue(np.all(np.eye(3) == rotation_matrix_from_vectors(vec1, vec1)))
        self.assertTrue(
            np.all(
                Rotation.from_matrix(rotation_matrix_from_vectors(vec1, vec2)).as_euler(
                    "xyz", degrees=True
                )
                == np.array([-90.0, 0.0, 0.0])
            )
        )
        with self.assertRaises(ValueError, msg="This should raise a ValueError") as _:
            rotation_matrix_from_vectors(err1, vec2)
        with self.assertRaises(ValueError, msg="This should raise a ValueError") as _:
            rotation_matrix_from_vectors(vec1, err2)
