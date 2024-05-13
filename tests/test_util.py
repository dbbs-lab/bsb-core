import unittest

import numpy as np
from bsb_test import (
    FixedPosConfigFixture,
    NumpyTestCase,
    RandomStorageFixture,
    skipIfOffline,
)
from scipy.spatial.transform import Rotation

from bsb import FileDependency, NeuroMorphoScheme, Scaffold
from bsb._util import assert_samelen, rotation_matrix_from_vectors


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


class TestUriSchemes(RandomStorageFixture, unittest.TestCase, engine_name="fs"):
    @skipIfOffline(scheme=NeuroMorphoScheme())
    def test_nm_scheme(self):
        file = FileDependency(
            "nm://AX2_scaled",
            Scaffold(storage=self.storage).files,
        )
        self.assertIs(NeuroMorphoScheme, type(file._scheme), "Expected NM scheme")
        meta = file.get_meta()
        self.assertIn("neuromorpho_data", meta)
        self.assertEqual(130892, meta["neuromorpho_data"]["neuron_id"])

    def test_nm_scheme_down(self):
        url = NeuroMorphoScheme._nm_url
        # Consistently trigger a 404 response in the NM scheme
        NeuroMorphoScheme._nm_url = "https://google.com/404"
        try:
            file = FileDependency(
                "nm://AX2_scaled",
                Scaffold(storage=self.storage).files,
            )
            with self.assertWarns(UserWarning) as w:
                file.get_meta()
        finally:
            NeuroMorphoScheme._nm_url = url


class TestAssertSameLength(unittest.TestCase):
    def test_same_length(self):
        assert_samelen([1, 2, 3], [4, 5, 6])
        with self.assertRaises(AssertionError):
            assert_samelen([1, 2], [2])
        assert_samelen([[1, 2]], [3])
        assert_samelen([], [])
