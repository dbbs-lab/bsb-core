import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold, from_storage
from bsb.config import from_json
from bsb.unittest import NetworkFixture, RandomStorageFixture, get_config_path


single_neuron_path = get_config_path("test_single_neuron.json")
multi_neuron_path = get_config_path("test_double_neuron.json")


class TestSingleTypeCompilation(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    """
    Check if we can compile a single cell type.
    """

    def setUp(self) -> None:
        self.cfg = from_json(single_neuron_path)
        super().setUp()
        self.network.compile()

    def test_cells_placed(self):
        self.assertEqual(
            4, len(self.network.get_placement_set("test_cell")), "should place 4 cells"
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
        self.cfg = from_json(multi_neuron_path)
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

    def test_redo_issue752(self):
        self.network.compile(redo=True, only=["connection"])
