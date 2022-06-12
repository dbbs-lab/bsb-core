import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold, from_hdf5
from bsb.config import from_json


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


single_neuron_config = relative_to_tests_folder("data/configs/test_single_neuron.json")


class TestSingleTypeCompilation(unittest.TestCase):
    """
    Check if the scaffold can create a single cell type.
    """

    @classmethod
    def setUpClass(self):
        super().setUpClass()
        config = from_json(single_neuron_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile(clear=True)

    def test_cells_placed(self):
        # Fixed count not predictable, see https://github.com/dbbs-lab/bsb/issues/489
        # self.assertEqual(4, len(self.scaffold.get_placement_set("test_cell")), 'count?')
        pass
