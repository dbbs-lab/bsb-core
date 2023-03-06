import unittest, os, sys, numpy as np, h5py, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


minimal_config_entities = relative_to_tests_folder("configs/test_minimal_entities.json")


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestEntities(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        import nest

        nest.ResetKernel()
        super(TestEntities, self).setUpClass()
        config = JSONConfig(file=minimal_config_entities)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        hdf_config = _from_storage("minimal_entities.hdf5")
        self.scaffold_fresh = Scaffold(hdf_config, from_file="minimal_entities.hdf5")

    def test_placement_statistics(self):
        self.assertEqual(self.scaffold.statistics.cells_placed["entity_type"], 100)

    @unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
    def test_creation_in_nest(self):
        f = h5py.File("minimal_entities.hdf5", "r")
        ids = list(f["entities"]["entity_type"])
        self.assertEqual(ids, list(range(100)))
        f.close()

        # Try to load the network directly from the hdf5 file
        nest_adapter = self.scaffold_fresh.get_simulation("test")
        simulator = nest_adapter.prepare()
        nest_adapter.simulate(simulator)
        nest_ids = nest_adapter.entities["entity_type"].nest_identifiers
        self.assertEqual(list(nest_ids), list(range(1, 101)))
