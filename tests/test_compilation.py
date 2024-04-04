import unittest

from bsb_test import NetworkFixture, RandomStorageFixture, get_test_config

from bsb import Configuration, InputError


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
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.cfg = Configuration.default(
            **{
                "name": "test",
                "partitions": {
                    "layer": {"thickness": 50.0, "stack_index": 0},
                },
                "cell_types": {
                    "cell": {
                        "spatial": {
                            "radius": 1,
                            "count": 1,
                        },
                    },
                },
                "placement": {
                    "layer_placement": {
                        "strategy": "bsb.placement.FixedPositions",
                        "partitions": ["layer"],
                        "cell_types": ["cell"],
                        "positions": [[0, 0, 0]],
                    }
                },
                "connectivity": {
                    "cell_to_cell": {
                        "strategy": "bsb.connectivity.AllToAll",
                        "presynaptic": {"cell_types": ["cell"]},
                        "postsynaptic": {"cell_types": ["cell"]},
                    },
                },
            }
        )
        super().setUp()

    def test_redo_issue752(self):
        # `redo` was untested
        self.network.compile(redo=True, only=["cell_to_cell"])

    def test_redo_issue763(self):
        # Test that users are protected against removing data by incorrect usage of
        # `append`/`redo`
        self.network.compile(clear=True)
        self.assertEqual(
            1,
            len(self.network.cell_types.cell.get_placement_set()),
            "test setup should place 1 cell",
        )
        with self.assertRaises(InputError, msg="should error incorrect usage"):
            self.network.compile(redo=["cell_to_cell"])
        self.network.compile(redo=True, only=["cell_to_cell"])
        self.assertEqual(
            1,
            len(self.network.cell_types.cell.get_placement_set()),
            "redoing a conn strat should not affect the placement",
        )
