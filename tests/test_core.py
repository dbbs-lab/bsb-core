import os
import unittest

from bsb_test import NetworkFixture, RandomStorageFixture

from bsb import core
from bsb.config import Configuration
from bsb.storage.interfaces import PlacementSet


class TestCore(RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="fs"):
    def setUp(self):
        self.cfg = Configuration.default()
        super().setUp()

    def test_from_storage(self):
        """Use the `from_storage` function to load a network."""
        self.network.compile(clear=True)
        core.from_storage(self.network.storage.root)

    def test_missing_storage(self):
        with self.assertRaises(FileNotFoundError):
            core.from_storage("does_not_exist")

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink("does_not_exist2")
        except Exception:
            pass

    def test_set_netw_root_nodes(self):
        """
        Test the anti-pattern of resetting the storage configuration for runtime errors.
        """
        self.network.storage_cfg = {"root": self.network.storage.root, "engine": "hdf5"}

    def test_set_netw_config(self):
        """
        Test resetting the configuration object
        """
        self.network.configuration = Configuration.default(
            regions=dict(x=dict(children=[]))
        )
        self.assertEqual(1, len(self.network.regions), "setting cfg failed")

    def test_netw_props(self):
        """
        Test the storage engine property keys like `.morphologies` and `.files`
        """
        self.assertEqual(
            0, len(self.network.morphologies.all()), "just checking morph prop"
        )

    def test_resize(self):
        self.network.partitions.add("layer", thickness=100)
        self.network.regions.add("region", children=["layer"])
        self.network.resize(x=500, y=500, z=500)
        self.assertEqual(500, self.network.network.x, "resize didnt update network node")
        self.assertEqual(
            500, self.network.partitions.layer.data.width, "didnt resize layer"
        )

    def test_get_placement_sets(self):
        """
        Test that placement sets for cell types are automatically initialized
        """
        self.network.cell_types.add("my_type", spatial=dict(radius=2, density=1))
        pslist = self.network.get_placement_sets()
        self.assertIsInstance(pslist, list, "should get list of PS")
        self.assertEqual(1, len(pslist), "should have one PS per cell type")
        self.assertIsInstance(pslist[0], PlacementSet, "elements should be PS")

    def test_diagrams(self):
        self.network.cell_types.add("cell1", {"spatial": {"radius": 1, "density": 1}})
        self.network.cell_types.add("cell2", {"spatial": {"radius": 1, "density": 1}})
        self.network.placement.add(
            "p1",
            {
                "strategy": "bsb.placement.FixedPositions",
                "positions": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "cell_types": ["cell1", "cell2"],
                "partitions": [],
            },
        )
        self.network.connectivity.add(
            "a_to_b",
            {
                "strategy": "bsb.connectivity.AllToAll",
                "presynaptic": {"cell_types": ["cell1"]},
                "postsynaptic": {"cell_types": ["cell2"]},
            },
        )
        cfg_diagram = self.network.get_config_diagram()
        self.assertIn('digraph "network"', cfg_diagram)
        self.assertIn('cell1[label="cell1"]', cfg_diagram)
        self.assertIn('cell2[label="cell2"]', cfg_diagram)
        self.assertIn('cell1 -> cell2[label="a_to_b"]', cfg_diagram)
        self.network.compile()
        storage_diagram = self.network.get_storage_diagram()
        self.assertIn('digraph "network"', storage_diagram)
        self.assertIn('cell1[label="cell1 (3 cell1)"]', storage_diagram)
        self.assertIn('cell2[label="cell2 (3 cell2)"]', storage_diagram)
        self.assertIn('cell1 -> cell2[label="a_to_b (9)"]', storage_diagram)


class TestProfiling(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.cfg = Configuration.default()
        super().setUp()

    def test_profiling(self):
        import bsb.profiling

        bsb.options.profiling = True
        self.network.compile()

        self.assertGreater(
            len(bsb.profiling.get_active_session()._meters), 0, "missing meters"
        )
        bsb.options.profiling = False
