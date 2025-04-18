import unittest

import numpy as np
from bsb_arbor.device import ArborDevice
from bsb_test import FixedPosConfigFixture, NumpyTestCase, RandomStorageFixture

from bsb import AdapterCheckpoint, Scaffold, config, get_simulation_adapter


class TestSimulate(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.cfg.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"]),
                postsynaptic=dict(cell_types=["test_cell"]),
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(clear=True)

    def test_simulate(self):
        self.network.simulations.add(
            "test",
            simulator="arbor",
            duration=100,
            resolution=1.0,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(),
        )
        self.network.run_simulation("test")


@config.node
class MockDevice(ArborDevice):
    def implement(self):
        pass

    def implement_probes(self, simdata, target):
        pass

    def implement_generators(self, simdata, target):
        pass


class test_adaptercheckpoint(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):

    def setUp(self):
        super().setUp()
        self.cfg.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"]),
                postsynaptic=dict(cell_types=["test_cell"]),
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(clear=True)
        self.network.simulations.add(
            "test",
            simulator="arbor",
            duration=100,
            resolution=0.25,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(
                test_mock={
                    "device": MockDevice,
                    "targetting": {"strategy": "all"},
                    "resolution": 0.25,
                    "checkpoints": [12.5, 25, 50],
                }
            ),
        )

    def test_wrong_value(self):
        """Check that checkpoints values are multiple of simulation resolution"""
        self.network.simulations.add(
            "wtest",
            simulator="arbor",
            duration=100,
            resolution=1,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(
                test_mock={
                    "device": MockDevice,
                    "targetting": {"strategy": "all"},
                    "resolution": 1,
                    "checkpoints": [12.5, 25, 50],
                }
            ),
        )
        sim = self.network.simulations["wtest"]
        AC = AdapterCheckpoint([sim])
        with self.assertRaises(ValueError):
            AC.suitable_step(1)

    def test_checkpoints(self):

        sim = self.network.simulations["test"]
        print(self.network.simulations["test"].simulator)
        AC = AdapterCheckpoint([sim])
        min_step = AC.suitable_step(1)
        self.assertEqual(
            min_step, 0.5, "Suitable step should lower the progression step from 1 to 0.5"
        )
        self.assertEqual(
            AC.sort_checkpoints(),
            [12.5, 25, 50],
            "Do not return the correct checkpoints list",
        )

    def test_multi_sim(self):
        self.network.simulations.add(
            "2_sim",
            simulator="arbor",
            duration=100,
            resolution=0.25,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(
                test_mock={
                    "device": MockDevice,
                    "targetting": {"strategy": "all"},
                    "resolution": 0.25,
                    "checkpoints": [17, 20, 25, 64, 71],
                }
            ),
        )
        sim = [self.network.simulations["test"], self.network.simulations["2_sim"]]
        AC = AdapterCheckpoint(sim)
        min_step = AC.suitable_step(1)
        self.assertEqual(
            min_step, 0.5, "Suitable step should lower the progression step from 1 to 0.5"
        )

        time_iterator = iter(
            np.arange(0, self.network.simulations["test"].duration, min_step)
        )
        check_points = []
        sim_ref = []
        for step in time_iterator:
            if AC.get_status(step):
                check_points.append(step)
                sim_ref.append([sim.name for sim in AC.checkpoints[step]])
        expected_sim_order = [
            ["test"],
            ["2_sim"],
            ["2_sim"],
            ["test", "2_sim"],
            ["test"],
            ["2_sim"],
            ["2_sim"],
        ]
        self.assertEqual(
            sim_ref,
            expected_sim_order,
            "The references to simulations are wrongly assigned",
        )
        self.assertEqual(
            check_points,
            [12.5, 17, 20, 25, 50, 64, 71],
            "Do not return the correct checkpoints list",
        )
