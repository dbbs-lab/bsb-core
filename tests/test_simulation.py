from bsb.core import Scaffold
from bsb.services import MPI
from bsb.config import Configuration
from bsb.morphologies import Morphology, Branch
from bsb.unittest import (
    NumpyTestCase,
    FixedPosConfigFixture,
    RandomStorageFixture,
    MorphologiesFixture,
    NetworkFixture,
    skip_parallel,
)
import unittest
import numpy as np
from collections import defaultdict


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
            simulator="neuron",
            duration=100,
            temperature=32,
            cell_models=dict(),
            connection_models=dict(),
            devices=dict(),
        )
        self.network.run_simulation("test")
