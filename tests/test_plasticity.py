import unittest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import from_hdf5

class TestNeuronCreation(unittest.TestCase):

    def setUp(self):
        config = from_hdf5("test_single_connection.hdf5")
        self.scaffold = Scaffold(config, from_file="test_single_connection.hdf5")
