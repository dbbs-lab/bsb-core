import importlib
import os
import unittest

from bsb.core import Scaffold


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


def reset_kernel(f):
    def reset(cls):
        import nest

        nest.ResetKernel()
        f(cls)

    return reset


minimal_config = relative_to_tests_folder("configs/test_minimal_simulation.json")
single_neuron_config = relative_to_tests_folder("configs/test_single_neuron.json")
recorder_config = relative_to_tests_folder("configs/test_recorders.json")
double_neuron_config = relative_to_tests_folder("configs/test_double_neuron.json")
double_nn_config = relative_to_tests_folder("configs/test_double_neuron_network.json")
homosyn_config = relative_to_tests_folder(
    "configs/test_double_neuron_network_homosyn.json"
)
heterosyn_config = relative_to_tests_folder(
    "configs/test_double_neuron_network_heterosyn.json"
)


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestKernelManagement(unittest.TestCase):
    # TODO: Add set_threads exception tests here
    pass


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestSingleNeuronTypeSetup(unittest.TestCase):
    def setUp(self):
        config = JSONConfig(file=single_neuron_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.configuration.simulations["test_single_neuron"]
        self.nest_adapter.reset()

    def tearDown(self):
        self.nest_adapter.delete_lock()

    def test_single_neuron(self):
        self.scaffold.run_simulation("test_single_neuron")
        test_cell_model = self.nest_adapter.cell_models["test_cell"]
        self.assertEqual(test_cell_model.nest_identifiers, list(range(1, 5)))

        test_neuron_status = self.nest_adapter.nest.GetStatus(
            test_cell_model.nest_identifiers
        )
        self.assertEqual(test_neuron_status[0]["t_ref"], 1.5)
        self.assertEqual(test_neuron_status[0]["C_m"], 7.0)
        self.assertEqual(test_neuron_status[0]["V_th"], -41.0)
        self.assertEqual(test_neuron_status[0]["V_reset"], -70.0)
        self.assertEqual(test_neuron_status[0]["E_L"], -62.0)
        self.assertEqual(test_neuron_status[0]["I_e"], 0.0)


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestDoubleNeuronTypeSetup(unittest.TestCase):
    @classmethod
    @reset_kernel
    def setUpClass(cls):
        super().setUpClass()
        config = JSONConfig(file=double_neuron_config)
        cls.scaffold = Scaffold(config)
        cls.scaffold.compile_network()
        cls.nest_adapter = cls.scaffold.configuration.simulations["test_double_neuron"]
        cls.scaffold.run_simulation("test_double_neuron")

    @classmethod
    def tearDownClass(cls):
        cls.nest_adapter.delete_lock()

    def test_double_neuron_creation(self):
        from_cell_model = self.nest_adapter.cell_models["from_cell"]
        self.assertEqual(from_cell_model.nest_identifiers, [1, 2, 3, 4])
        to_cell_model = self.nest_adapter.cell_models["to_cell"].nest_identifiers
        self.assertEqual(to_cell_model, [5, 6, 7, 8])

    def test_double_neuron_iaf_params(self):
        from_cell_model = self.nest_adapter.cell_models["from_cell"]
        from_neuron_status = self.nest_adapter.nest.GetStatus(
            from_cell_model.nest_identifiers
        )
        self.assertEqual(from_neuron_status[0]["t_ref"], 1.5)
        self.assertEqual(from_neuron_status[0]["C_m"], 7.0)
        self.assertEqual(from_neuron_status[0]["V_th"], -41.0)
        self.assertEqual(from_neuron_status[0]["V_reset"], -70.0)
        self.assertEqual(from_neuron_status[0]["E_L"], -62.0)
        self.assertEqual(from_neuron_status[0]["I_e"], 0.0)

    def test_double_neuron_eglif_params(self):
        to_cell_model = self.nest_adapter.cell_models["to_cell"]
        to_neuron_status = self.nest_adapter.nest.GetStatus(
            to_cell_model.nest_identifiers
        )
        self.assertEqual(to_neuron_status[0]["t_ref"], 1.5)
        self.assertEqual(to_neuron_status[0]["C_m"], 7.0)
        self.assertEqual(to_neuron_status[0]["V_th"], -41.0)
        self.assertEqual(to_neuron_status[0]["V_reset"], -70.0)
        self.assertEqual(to_neuron_status[0]["E_L"], -62.0)
        self.assertEqual(to_neuron_status[0]["I_e"], -0.888)
        self.assertEqual(to_neuron_status[0]["Vinit"], -62.0)
        self.assertEqual(to_neuron_status[0]["lambda_0"], 1.0)
        self.assertEqual(to_neuron_status[0]["tau_V"], 0.3)
        self.assertEqual(to_neuron_status[0]["kadap"], 0.022)
        self.assertEqual(to_neuron_status[0]["k1"], 0.311)
        self.assertEqual(to_neuron_status[0]["k2"], 0.041)
        self.assertEqual(to_neuron_status[0]["A1"], 0.01)
        self.assertEqual(to_neuron_status[0]["A2"], -0.94)


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestDoubleNeuronNetworkStatic(unittest.TestCase):
    @classmethod
    @reset_kernel
    def setUpClass(cls):
        super().setUpClass()
        config = JSONConfig(file=double_nn_config)
        if not neuron_installed():
            del config.simulations["neuron"]
        cls.scaffold = Scaffold(config)
        cls.scaffold.compile_network()
        cls.nest_adapter = cls.scaffold.run_simulation(
            "test_double_neuron_network_static"
        )

    @classmethod
    def tearDownClass(cls):
        cls.nest_adapter.delete_lock()

    def test_double_neuron_network(self):
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        self.assertIsNotNone(conn)

    def test_double_neuron_network_params(self):
        import numpy as np

        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        self.assertEqual(
            self.nest_adapter.nest.GetStatus(conn, "weight"),
            tuple(9.0 * np.ones(len(conn))),
        )
        self.assertEqual(
            self.nest_adapter.nest.GetStatus(conn, "delay"),
            tuple(4.0 * np.ones(len(conn))),
        )


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestDoubleNeuronNetworkHomosyn(unittest.TestCase):
    @classmethod
    @reset_kernel
    def setUpClass(cls):
        super().setUpClass()
        config = JSONConfig(file=homosyn_config)
        cls.scaffold = Scaffold(config)
        cls.scaffold.compile_network()
        cls.nest_adapter = cls.scaffold.run_simulation(
            "test_double_neuron_network_homosyn"
        )

    @classmethod
    def tearDownClass(cls):
        cls.nest_adapter.delete_lock()

    def test_double_neuron_network(self):
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        self.assertIsNotNone(conn)

    def test_double_neuron_network_plasticity(self):
        import numpy as np

        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        # Verify that weights re changing
        self.assertNotEqual(
            self.nest_adapter.nest.GetStatus(conn, "weight"),
            tuple(9.0 * np.ones(len(conn))),
        )
        self.assertEqual(
            self.nest_adapter.nest.GetStatus(conn, "delay"),
            tuple(4.0 * np.ones(len(conn))),
        )


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestDoubleNeuronNetworkHeterosyn(unittest.TestCase):
    @classmethod
    @reset_kernel
    def setUpClass(cls):
        super().setUpClass()
        config = JSONConfig(file=heterosyn_config)
        cls.scaffold = Scaffold(config)
        cls.scaffold.compile_network()
        cls.nest_adapter = cls.scaffold.run_simulation(
            "test_double_neuron_network_heterosyn"
        )

    @classmethod
    def tearDownClass(cls):
        cls.nest_adapter.delete_lock()

    def test_double_neuron_network(self):
        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        self.assertIsNotNone(conn)
        teaching_cell_model = self.nest_adapter.cell_models["teaching_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn_teaching = self.nest_adapter.nest.GetConnections(
            teaching_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        self.assertIsNotNone(conn_teaching)

    def test_double_neuron_network_plasticity(self):
        import numpy as np

        source_cell_model = self.nest_adapter.cell_models["from_cell"]
        target_cell_model = self.nest_adapter.cell_models["to_cell"]
        conn = self.nest_adapter.nest.GetConnections(
            source_cell_model.nest_identifiers, target_cell_model.nest_identifiers
        )
        # Verify that weights re changing
        self.assertNotEqual(
            self.nest_adapter.nest.GetStatus(conn, "weight"),
            tuple(9.0 * np.ones(len(conn))),
        )
        self.assertEqual(
            self.nest_adapter.nest.GetStatus(conn, "delay"),
            tuple(4.0 * np.ones(len(conn))),
        )

    def test_teaching_connection_missing(self):
        from bsb.exceptions import ConfigurationError

        with open(heterosyn_config, "r") as f:
            stream = f.read()
        stream = stream.replace('"teaching": "teaching_cell_to_cell",', "")

        with self.assertRaises(ConfigurationError) as ce:
            config = JSONConfig(stream=stream)
            self.scaffold = Scaffold(config)

    def test_teaching_connection_configuration(self):
        from bsb.exceptions import ConfigurationError

        with open(heterosyn_config, "r") as f:
            stream = f.read()
        stream = stream.replace(
            '"teaching": "teaching_cell_to_cell",', '"teaching": "random_connection",'
        )

        with self.assertRaises(ConfigurationError) as ce:
            config = JSONConfig(stream=stream)
            self.scaffold = Scaffold(config)


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestMultiInstance(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestMultiInstance, self).setUpClass()
        import nest

        self.nest = nest
        config = JSONConfig(file=single_neuron_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.hdf5 = self.scaffold.output_formatter.file
        self.nest_adapter_0 = self.scaffold.get_simulation("test_single_neuron")
        # When another test errors, the lock might remain, and all locking tests fail
        self.nest_adapter_0.delete_lock()
        self.nest_adapter_1 = self.scaffold.create_adapter("test_single_neuron")
        self.nest_adapter_2 = self.scaffold.create_adapter("test_single_neuron")
        self.nest_adapter_multi_1 = self.scaffold.create_adapter("test_single_neuron")
        self.nest_adapter_multi_1.enable_multi("first")
        self.nest_adapter_multi_1b = self.scaffold.create_adapter("test_single_neuron")
        self.nest_adapter_multi_1b.enable_multi("first")
        self.nest_adapter_multi_2 = self.scaffold.create_adapter("test_single_neuron")
        self.nest_adapter_multi_2.enable_multi("second")

    def tearDown(self):
        # Clean up any remaining locks to keep the test functions independent.
        # Otherwise a chain reaction of failures is evoked.
        self.nest_adapter_0.delete_lock()

    def test_single_instance_unwanted_usage(self):
        # Test AdapterError when trying to unlock unlocked adapter
        self.assertRaises(AdapterError, self.nest_adapter_0.release_lock)
        # Test whether the scaffold throws an AdapterError when the same
        # adapter is prepared twice.
        self.nest_adapter_0.prepare()
        self.assertRaises(AdapterError, self.nest_adapter_0.prepare)

        self.nest_adapter_0.release_lock()
        self.nest_adapter_0.reset()

    def test_single_instance_single_lock(self):
        self.nest_adapter_1.reset()
        # Lock kernel. Prepare adapter and thereby lock NEST kernel
        self.nest_adapter_1.prepare()
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
        self.nest_adapter_1.prepare()

        self.assertRaises(KernelLockedError, self.nest_adapter_2.prepare)
        self.assertEqual(self.nest_adapter_2.is_prepared, False)

        self.nest_adapter_1.release_lock()
        self.nest_adapter_1.reset()
        self.nest_adapter_2.reset()

    def test_single_instance_multi_lock(self):
        self.nest_adapter_multi_1.reset()
        # Test functionality of the multi lock.
        self.nest_adapter_multi_1.prepare()
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
        self.nest_adapter_multi_1.prepare()
        # Test that we have 1 lock.
        lock_data = self.nest_adapter_multi_1.read_lock()
        # Check multi instance multi lock
        self.nest_adapter_multi_2.cell_models["test_cell"].parameters["t_ref"] = 3.0
        self.nest_adapter_multi_2.prepare()
        # Check that we have 2 locks
        lock_data = self.nest_adapter_multi_1.read_lock()
        self.assertEqual(len(lock_data["suffixes"]), 2)

        # Test that we set the right parameters on the right classes.
        try:
            params = self.nest.GetDefaults("test_cell_first")
        except Exception as e:
            self.fail("First suffixed NEST models not found")
        try:
            params = self.nest.GetDefaults("test_cell_second")
        except Exception as e:
            self.fail("Second suffixed NEST models not found")

        # Test that the adapters have the correct nest_identifiers
        id1 = self.nest_adapter_multi_1.cell_models["test_cell"].nest_identifiers
        id2 = self.nest_adapter_multi_2.cell_models["test_cell"].nest_identifiers
        self.assertEqual(id1, [1, 2, 3, 4])
        self.assertEqual(id2, [5, 6, 7, 8])

        # Test that the adapter nodes have the right model
        self.assertTrue(
            all(
                map(
                    lambda x: str(x["model"]) == "test_cell_first",
                    self.nest.GetStatus(id1),
                )
            )
        )
        self.assertTrue(
            all(
                map(
                    lambda x: str(x["model"]) == "test_cell_second",
                    self.nest.GetStatus(id2),
                )
            )
        )

        # Test that the adapter nodes have the right differential parameter t_ref
        self.assertTrue(all(map(lambda x: x["t_ref"] == 1.5, self.nest.GetStatus(id1))))
        self.assertTrue(all(map(lambda x: x["t_ref"] == 3.0, self.nest.GetStatus(id2))))

        # Check duplicate suffixes
        self.assertRaises(SuffixTakenError, self.nest_adapter_multi_1b.prepare)

        self.nest_adapter_multi_1.release_lock()
        self.nest_adapter_multi_1.reset()
        # Test that we have 1 lock again after release.
        lock_data = self.nest_adapter_multi_1.read_lock()
        self.assertEqual(lock_data["suffixes"][0], "second")
        self.nest_adapter_multi_2.release_lock()
        self.nest_adapter_multi_2.reset()


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(importlib.util.find_spec("nest") is None, "NEST is not importable.")
class TestDeviceProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super().setUpClass()
        import nest

        self.nest = nest
        config = JSONConfig(file=recorder_config)
        self.scaffold = Scaffold(config)

    def test_interface(self):
        import bsb.simulators.nest

        adapter = self.scaffold.configuration.simulations["test_recorders"]
        sp = bsb.simulators.nest.get_device_protocol(adapter.devices["record_spikes"])
        gen = bsb.simulators.nest.get_device_protocol(adapter.devices["gen"])
        self.assertEqual(bsb.simulators.nest.DeviceProtocol, gen.__class__)
        self.assertEqual(bsb.simulators.nest.SpikeDetectorProtocol, sp.__class__)

    def test_spike_recorder(self):
        adapter = self.scaffold.configuration.simulations["test_recorders"]
        self.assertEqual(0, len(adapter.result.recorders))
        simulator = adapter.prepare()
        self.assertEqual(1, len(adapter.result.recorders))
        adapter.simulate(simulator)
        adapter.collect_output()
