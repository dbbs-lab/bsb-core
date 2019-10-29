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
        self.assertEqual(test_cell_model.identifiers, list(range(1,5)))

    #def test_single_neuron_parameters(self):    # Parameters
        test_neuron_status = self.nest_adapter.nest.GetStatus(test_cell_model.identifiers)
        self.assertEqual(test_neuron_status[0]['t_ref'], 1.5)
        self.assertEqual(test_neuron_status[0]['C_m'], 7.0)
        self.assertEqual(test_neuron_status[0]['V_th'], -41.0)
        self.assertEqual(test_neuron_status[0]['V_reset'], -70.0)
        self.assertEqual(test_neuron_status[0]['E_L'], -62.0)
        self.assertEqual(test_neuron_status[0]['I_e'], 0.0)


class TestDoubleNeuronTypeSetup(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronTypeSetup, self).setUpClass()
        config = JSONConfig(file="test_double_neuron.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations['test_double_neuron']

    def test_double_neuron_creation(self):
        self.scaffold.run_simulation("test_double_neuron")
        from_cell_model = self.nest_adapter.cell_models["from_cell"]
        self.assertEqual(from_cell_model.identifiers, [1, 2, 3, 4])
        to_cell_model = self.nest_adapter.cell_models["to_cell"].identifiers
        self.assertEqual(to_cell_model, [5, 6, 7, 8])

    def test_double_neuron_iaf_params(self):
        from_cell_model = self.nest_adapter.cell_models["from_cell"]
        from_neuron_status = self.nest_adapter.nest.GetStatus(from_cell_model.identifiers)
        self.assertEqual(from_neuron_status[0]['t_ref'], 1.5)
        self.assertEqual(from_neuron_status[0]['C_m'], 7.0)
        self.assertEqual(from_neuron_status[0]['V_th'], -41.0)
        self.assertEqual(from_neuron_status[0]['V_reset'], -70.0)
        self.assertEqual(from_neuron_status[0]['E_L'], -62.0)
        self.assertEqual(from_neuron_status[0]['I_e'], 0.0)

    def test_double_neuron_eglif_params(self):
        to_cell_model = self.nest_adapter.cell_models["to_cell"]
        to_neuron_status = self.nest_adapter.nest.GetStatus(to_cell_model.identifiers)
        self.assertEqual(to_neuron_status[0]['t_ref'], 1.5)
        self.assertEqual(to_neuron_status[0]['C_m'], 7.0)
        self.assertEqual(to_neuron_status[0]['Vth_init'], -41.0)
        self.assertEqual(to_neuron_status[0]['V_reset'], -70.0)
        self.assertEqual(to_neuron_status[0]['E_L'], -62.0)
        self.assertEqual(to_neuron_status[0]['Ie_const'], -0.888)
        self.assertEqual(to_neuron_status[0]['Vinit'], -62.0)
        self.assertEqual(to_neuron_status[0]['lambda_0'], 1.0)
        self.assertEqual(to_neuron_status[0]['delta_V'], 0.3)
        self.assertEqual(to_neuron_status[0]['Ie_const'], -0.888)
        self.assertEqual(to_neuron_status[0]['adaptC'], 0.022)
        self.assertEqual(to_neuron_status[0]['k1'], 0.311)
        self.assertEqual(to_neuron_status[0]['k2'], 0.041)
        self.assertEqual(to_neuron_status[0]['A1'], 0.01)
        self.assertEqual(to_neuron_status[0]['A2'], -0.94)


class TestDoubleNeuronNetworkStatic(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronNetworkStatic, self).setUpClass()
        config = JSONConfig(file="test_double_neuron_network.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations['test_double_neuron_network_static']


    def test_double_neuron_network(self):
        self.scaffold.run_simulation("test_double_neuron_network_static")
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(source_cell_model.identifiers,target_cell_model.identifiers)
        self.assertIsNotNone(conn)

    def test_double_neuron_network_params(self):
        import numpy as np
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(source_cell_model.identifiers,target_cell_model.identifiers)
        self.assertEqual(self.nest_adapter.nest.GetStatus(conn,"weight"), tuple(-9.0*np.ones(len(conn))))
        self.assertEqual(self.nest_adapter.nest.GetStatus(conn,"delay"), tuple(4.0*np.ones(len(conn))))



class TestDoubleNeuronNetworkHomosyn(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronNetworkHomosyn, self).setUpClass()
        config = JSONConfig(file="test_double_neuron_network_homosyn.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations['test_double_neuron_network_homosyn']


    def test_double_neuron_network(self):
        self.scaffold.run_simulation("test_double_neuron_network_homosyn")
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(source_cell_model.identifiers,target_cell_model.identifiers)
        self.assertIsNotNone(conn)

    def test_double_neuron_network_plasticity(self):
        import numpy as np
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(source_cell_model.identifiers,target_cell_model.identifiers)
        print((self.nest_adapter.nest.GetStatus(conn,"weight")))
        self.assertEqual(self.nest_adapter.nest.GetStatus(conn,"weight"), tuple(9.0*np.ones(len(conn))))
