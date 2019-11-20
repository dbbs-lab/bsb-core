import unittest, os, sys, numpy as np, h5py, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.simulators.nest import NestCell
from scaffold.models import Layer, CellType
from scaffold.exceptions import AdapterException, KernelLockedException

def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)

minimal_config = relative_to_tests_folder("configs/test_minimal_simulation.json")
single_neuron_config = relative_to_tests_folder("configs/test_single_neuron.json")
double_neuron_config = relative_to_tests_folder("configs/test_double_neuron.json")
double_nn_config = relative_to_tests_folder("configs/test_double_neuron_network.json")
homosyn_config = relative_to_tests_folder("configs/test_double_neuron_network_homosyn.json")

@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestKernelManagement(unittest.TestCase):
    # TODO: Add set_threads exception tests here
    pass


@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestSingleNeuronTypeSetup(unittest.TestCase):

    def setUp(self):
        config = JSONConfig(file=single_neuron_config)
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


@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestDoubleNeuronTypeSetup(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronTypeSetup, self).setUpClass()
        config = JSONConfig(file=double_neuron_config)
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


@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestDoubleNeuronNetworkStatic(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronNetworkStatic, self).setUpClass()
        config = JSONConfig(file=double_nn_config)
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


@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestDoubleNeuronNetworkHomosyn(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestDoubleNeuronNetworkHomosyn, self).setUpClass()
        config = JSONConfig(file=homosyn_config)
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
        # Verify that weights re changing
        self.assertNotEqual(self.nest_adapter.nest.GetStatus(conn,"weight"), tuple(9.0*np.ones(len(conn))))
        self.assertEqual(self.nest_adapter.nest.GetStatus(conn,"delay"), tuple(4.0*np.ones(len(conn))))


@unittest.skipIf(importlib.find_loader('nest') is None, "NEST is not importable.")
class TestMultiInstance(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestMultiInstance, self).setUpClass()
        config = JSONConfig(file=minimal_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.hdf5 = self.scaffold.output_formatter.file
        self.nest_adapter_0 = self.scaffold.get_simulation('test')
        self.nest_adapter_1 = self.scaffold.create_adapter('test')
        self.nest_adapter_2 = self.scaffold.create_adapter('test')
        self.nest_adapter_multi_1 = self.scaffold.create_adapter('test')
        self.nest_adapter_multi_1.enable_multi('first')
        self.nest_adapter_multi_1b = self.scaffold.create_adapter('test')
        self.nest_adapter_multi_1b.enable_multi('first')
        self.nest_adapter_multi_2 = self.scaffold.create_adapter('test')
        self.nest_adapter_multi_2.enable_multi('second')

    def test_single_instance_unwanted_usage(self):
        # Test AdapterException when trying to unlock unlocked adapter
        self.assertRaises(AdapterException, self.nest_adapter_0.release_lock)
        # Test whether the scaffold throws an AdapterException when the same
        # adapter is prepared twice.
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_0.prepare(handle)

        with h5py.File(self.hdf5, "r") as handle:
            self.assertRaises(AdapterException, self.nest_adapter_0.prepare, handle)

        self.nest_adapter_0.reset()

    def test_single_instance_single_lock(self):
        # Lock kernel. Prepare adapter and thereby lock NEST kernel
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_1.prepare(handle)
        lock_data = self.nest_adapter_1.read_lock()
        self.assertEqual(lock_data["multi"], False)
        self.assertEqual(self.nest_adapter_1.multi, False)
        self.assertEqual(self.nest_adapter_1.has_lock, True)

        # Release lock.
        self.nest_adapter_1.release_lock()
        self.assertEqual(self.nest_adapter_1.read_lock(), None)
        self.assertEqual(self.nest_adapter_1.has_lock, False)
        self.nest_adapter_1.reset()

    def test_multi_instance_single_lock(self):
        # Test that a 2nd single-instance adapter can't manage a locked kernel.
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_1.prepare(handle)

        with h5py.File(self.hdf5, "r") as handle:
            self.assertRaises(KernelLockedException, self.nest_adapter_2.prepare, handle)
        self.assertEqual(self.nest_adapter_2.is_prepared, False)

        self.nest_adapter_1.release_lock()
        self.nest_adapter_1.reset()
        self.nest_adapter_2.reset()

    def test_single_instance_multi_lock(self):
        # Test functionality of the multi lock.
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_multi_1.prepare(handle)
        lock_data = self.nest_adapter_multi_1.read_lock()
        self.assertEqual(self.nest_adapter_multi_1.suffix, "first")
        self.assertEqual(lock_data["multi"], True)
        self.assertEqual(lock_data["suffixes"][0], self.nest_adapter_multi_1.suffix)
        self.assertEqual(self.nest_adapter_multi_1.multi, True)
        self.assertEqual(self.nest_adapter_multi_1.is_prepared, True)
        self.assertEqual(self.nest_adapter_multi_1.has_lock, True)

        self.nest_adapter_multi_1.release_lock()
        self.assertEqual(self.nest_adapter_multi_1.multi, True)
        self.assertEqual(self.nest_adapter_multi_1.has_lock, False)
        self.nest_adapter_multi_1.reset()
        self.assertEqual(self.nest_adapter_multi_1.is_prepared, False)

    def test_multi_instance_multi_lock(self):
        # Test functionality of the multi lock.
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_multi_1.prepare(handle)
        # Check multi instance multi lock
        with h5py.File(self.hdf5, "r") as handle:
            self.nest_adapter_multi_2.prepare(handle)

        # Check duplicate suffixes
        with h5py.File(self.hdf5, "r") as handle:
            self.assertRaises(AdapterException, self.nest_adapter_multi_1b.prepare, handle)

        lock_data = self.nest_adapter_multi_1.read_lock()
        self.assertEqual(len(lock_data["suffixes"]), 2)

        self.nest_adapter_multi_1.release_lock()
        self.nest_adapter_multi_1.reset()
        self.nest_adapter_multi_2.release_lock()
        self.nest_adapter_multi_2.reset()
