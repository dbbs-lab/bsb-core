import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.simulators.nest import NestCell
from scaffold.models import Layer, CellType

class TestSingleNeuronTypeSetup(unittest.TestCase):

    def setUp(self):
        config = JSONConfig(file="test_single_neuron.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations['test_single_neuron']


    def test_single_neuron(self):
        self.scaffold.run_simulation("test_single_neuron")
        test_cell_model = self.nest_adapter.cell_models["test_cell"]
        self.assertEqual(test_cell_model.identifiers, [1, 2, 3, 4])


class TestDoubleNeuronTypeSetup(unittest.TestCase):

    def setUp(self):
        config = JSONConfig(file="test_double_neuron.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations['test_double_neuron']

    def test_double_neuron(self):
        self.scaffold.run_simulation("test_double_neuron")
        test_cell_model = self.nest_adapter.cell_models["from_cell"]
        self.assertEqual(test_cell_model.identifiers, [1, 2, 3, 4])
        test_cell_model = self.nest_adapter.cell_models["to_cell"]
        self.assertEqual(test_cell_model.identifiers, [5, 6, 7, 8])
