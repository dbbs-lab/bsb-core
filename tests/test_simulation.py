from bsb.core import Scaffold
from bsb.unittest import (
    NumpyTestCase,
    FixedPosConfigFixture,
    RandomStorageFixture,
)
import unittest


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
            simulator="nest",
            duration=100,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(),
        )
        self.network.run_simulation("test")
