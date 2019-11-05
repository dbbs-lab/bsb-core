import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.models import Layer, CellType

class TestSingleTypeCompilation(unittest.TestCase):
    '''
    Check if the scaffold can create a single cell type.
    '''

    @classmethod
    def setUpClass(self):
        super(TestSingleTypeCompilation, self).setUpClass()
        config = JSONConfig(file="configs/test_single_neuron.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()

    def test_placement_statistics(self):
        self.assertEqual(self.scaffold.statistics.cells_placed["test_cell"], 4)

    def test_network_cache(self):
        pass
        # TODO: Implement a check that the network cache contains the right amount of placed cells

    def test_hdf5_cells(self):
        pass
        # TODO: Implement a check that the hdf5 file contains the right datasets under 'cells'
