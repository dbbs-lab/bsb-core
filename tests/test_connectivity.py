import unittest, os, sys, numpy as np, h5py, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import Configuration


class TestAllToAll(unittest.TestCase):
    def setUp(self):
        super().setUpClass()
        cfg = Configuration.default()
        cfg.storage.root = "all_to_all.hdf5"
        cfg.regions.add("test_region")
        cfg.partitions.add("test_part", region="test_region", thickness=100)
        self.network = Scaffold(cfg)

    def test_one_type(self):
        t = self.network.cell_types.add("test", spatial=dict(radius=1, density=1e-3))
        fixed_place = self.network.placement.add(
            "test_place",
            cls="bsb.placement.FixedPositions",
            partitions=["test_part"],
            cell_types=["test"],
        )
        fixed_place.positions = np.tile(np.arange(10).reshape(-1, 1), 3)

        # Adding the conn strat, changes the placement data?

        # all_to_all = self.network.connectivity.add("test_all", cls="bsb.connectivity.AllToAll", presynaptic=dict(cell_types=["test"]), postsynaptic=dict(cell_types=["test"]))
        self.network.compile(clear=True)
        wanted = len(fixed_place.positions)
        print(t.get_placement_set().load_positions())
        placed = len(t.get_placement_set())
        self.assertEqual(wanted, placed, "incorrect num of cells placed")
