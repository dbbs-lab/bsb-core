import unittest

import numpy as np
from bsb_test import (
    NetworkFixture,
    NumpyTestCase,
    RandomStorageFixture,
    get_test_config,
)

from bsb import Configuration, InputError, RedoError


class TestSingleTypeCompilation(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    """
    Check if we can compile a single cell type.
    """

    def setUp(self) -> None:
        self.cfg = get_test_config("single")
        super().setUp()
        self.network.compile()

    def test_cells_placed(self):
        self.assertEqual(
            40, len(self.network.get_placement_set("test_cell")), "should place 40 cells"
        )
        self.assertNotEqual(
            0, len(self.network.get_placement_set("test_cell")), "No cells placed"
        )


class TestMultiTypeCompilation(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    """
    Check if we can compile several types, connected together
    """

    def setUp(self) -> None:
        self.cfg = get_test_config("double_neuron")
        super().setUp()
        self.network.compile()

    def test_multi_celltypes(self):
        ps_from = self.network.get_placement_set("from_cell")
        ps_to = self.network.get_placement_set("to_cell")
        self.assertEqual(4, len(ps_from), "should place 4 cells")
        self.assertEqual(4, len(ps_to), "should place 4 cells")
        csets = self.network.get_connectivity_sets()
        self.assertEqual(1, len(csets), "expected a connectivity set")
        cs = csets[0]
        self.assertEqual(
            cs.pre_type,
            self.network.cell_types.from_cell,
            "expected from_cell as presyn ct",
        )
        self.assertEqual(
            cs.post_type,
            self.network.cell_types.to_cell,
            "expected from_cell as presyn ct",
        )
        self.assertEqual(16, len(cs), "alltoall => 4x4 = 16")


class TestRedoCompilation(
    RandomStorageFixture,
    NetworkFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        self.cfg = Configuration.default(
            **{
                "name": "test",
                "network": {"chunk_size": [200, 200, 200]},
                "partitions": {
                    "layer": {"thickness": 50.0},
                },
                "cell_types": {
                    "cell": {
                        "spatial": {
                            "radius": 1,
                            "count": 2,
                        },
                    },
                    "cell2": {
                        "spatial": {
                            "radius": 1,
                            "count_ratio": 1.0,
                            "relative_to": "cell",
                        }
                    },
                },
                "placement": {
                    "layer_placement": {
                        "strategy": "bsb.placement.FixedPositions",
                        "partitions": ["layer"],
                        "cell_types": ["cell"],
                        "positions": [[0, 0, 0], [0, 0, 1]],
                    },
                    "second_placement": {
                        "strategy": "bsb.placement.RandomPlacement",
                        "partitions": ["layer"],
                        "cell_types": ["cell"],
                    },
                    "third_placement": {
                        "strategy": "bsb.placement.RandomPlacement",
                        "partitions": ["layer"],
                        "cell_types": ["cell2"],
                    },
                },
                "connectivity": {
                    "cell_to_cell": {
                        "strategy": "bsb.connectivity.AllToAll",
                        "presynaptic": {"cell_types": ["cell"]},
                        "postsynaptic": {"cell_types": ["cell"]},
                        "affinity": 0.5,
                    },
                },
            }
        )
        super().setUp()

    def test_redo_issue763(self):
        # Test that users are protected against removing data by incorrect usage of
        # `append`/`redo`
        self.network.compile(clear=True)
        ps = self.network.cell_types.cell.get_placement_set()
        self.assertEqual(
            4,
            len(ps),
            "test setup should place 4 cells",
        )
        positions = ps.load_positions()
        connections = self.network.get_connectivity_set("cell_to_cell")
        self.assertGreaterEqual(
            4 * 4,
            len(connections),
            "test setup should create 4 connections",
        )
        connections = np.array(connections.load_connections().all())
        with self.assertRaises(InputError, msg="should error incorrect usage"):
            self.network.compile(redo=["cell_to_cell"])
        self.network.compile(redo=True, only=["cell_to_cell"])
        ps = self.network.cell_types.cell.get_placement_set()
        self.assertEqual(
            4,
            len(ps),
            "redoing a conn strat should not affect the placement",
        )
        self.assertClose(positions, ps.load_positions(), atol=1e-5)
        new_connections = self.network.get_connectivity_set("cell_to_cell")
        self.assertGreaterEqual(
            4 * 4,
            len(new_connections),
            "redoing a conn strat should not duplicate the connections",
        )
        new_connections = np.array(new_connections.load_connections().all())
        # It is very unlikely in this configuration that
        # two random alltoall connectivity will generate the same result
        self.assertTrue(
            new_connections.size != connections.size
            or np.any(new_connections != connections)
        )

    def test_redo_placement(self):
        self.network.compile(clear=True)
        positions = self.network.cell_types.cell.get_placement_set().load_positions()
        positions2 = self.network.cell_types.cell2.get_placement_set().load_positions()
        connections = np.array(
            self.network.get_connectivity_set("cell_to_cell").load_connections().all()
        )
        # test redo placement affecting only cell2
        self.network.compile(redo=True, only=["third_placement"])
        self.assertClose(
            positions,
            self.network.cell_types.cell.get_placement_set().load_positions(),
            "Redoing placement on cell2 should not affect cell1",
            atol=1e-5,
        )
        self.assertClose(
            connections,
            np.array(
                self.network.get_connectivity_set("cell_to_cell").load_connections().all()
            ),
            "Redoing placement on cell2 should not affect cell1 connectivity",
            atol=1e-5,
        )
        new_positions = self.network.cell_types.cell2.get_placement_set().load_positions()
        self.assertAll(positions2 != new_positions)
        positions2 = np.copy(positions)
        # test redo layer_placement should affect everything else
        # since second_placement is also on cell1, third_placement depends on second_placement
        # and cell_to_cell is affected by cell1 placement
        self.network.compile(redo=True, only=["layer_placement"])
        self.assertAll(
            positions[:-2]
            != self.network.cell_types.cell.get_placement_set().load_positions()[:-2]
        )
        self.assertAll(
            positions2
            != self.network.cell_types.cell2.get_placement_set().load_positions()
        )
        new_connections = np.array(
            self.network.get_connectivity_set("cell_to_cell").load_connections().all()
        )
        # It is very unlikely in this configuration that
        # two random alltoall connectivity will generate the same result
        self.assertTrue(
            new_connections.size != connections.size
            or np.any(new_connections != connections)
        )

    def test_redo_skip_error(self):
        self.network.compile(clear=True)
        with self.assertRaises(RedoError):
            self.network.compile(
                redo=True, only=["layer_placement"], skip=["second_placement"]
            )
        self.network.compile(
            redo=True, only=["layer_placement"], skip=["second_placement"], force=True
        )
