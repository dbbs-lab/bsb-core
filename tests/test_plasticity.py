import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import ScaffoldConfig
from scaffold.simulators.nest import NestCell
from scaffold.models import Layer, CellType

class TestSingleNeuronSetup(unittest.TestCase):

    def setUp(self):
        config = ScaffoldConfig()
        self.scaffold = Scaffold(config)
        # Create fake layer
        test_layer = Layer("test_layer",[0.,0.,0.],[100.,100.,100.])
        self.scaffold.configuration.add_layer(test_layer)
        # Create fake cell type
        test_cell_type = CellType("test_type")
        test_cell_type.placement = type("placement", (object,), {})()
        test_cell_type.id = 0
        self.scaffold.configuration.add_cell_type(test_cell_type)
        self.scaffold.cells_by_type["test_type"] = np.empty((0, 5))
        self.scaffold.cells_by_layer["test_layer"] = np.empty((0, 5))
        # Place single cell
        self.scaffold.place_cells(test_cell_type,test_layer,np.array([[0.,0.,0.]]))
        nest_adapter = self.scaffold.simulators["nest"]()
        nest_adapter.duration = 10
        nest_adapter.initialise(self.scaffold)
        test_cell_model = NestCell(nest_adapter)
        test_cell_model.name = "test_type"
        test_cell_model.parameters = {
            "t_ref": 1.5,
            "C_m": 3.0,
            "V_th": -42.0,
            "V_reset": -84.0,
            "E_L": -74.0,
            "tau_syn_ex": 0.5,
            "tau_syn_in": 10.0,
            "g_L": 1.5
        }
        test_cell_model.iaf_cond_alpha = {
            "I_e": 0.0
        }
        test_cell_model.initialise(self.scaffold)
        test_cell_model.neuron_model = "iaf_cond_alpha"
        nest_adapter.cell_models[test_cell_model.name] = test_cell_model
        self.nest_adapter = nest_adapter


    def test_single_neuron(self):
        # self.nest_adapter.create_neurons(self.nest_adapter.cell_models)
        with h5py.File("test_single_connection.hdf5") as handle:
            self.nest_adapter.prepare(handle)
        test_cell_model = self.nest_adapter.cell_models["test_type"]
        self.assertEqual(test_cell_model.identifiers, [1])
