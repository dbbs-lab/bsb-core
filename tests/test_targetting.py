import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import JSONConfig
from scaffold.core import Scaffold
import scaffold.helpers
from scaffold.exceptions import *


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


double_nn_config = relative_to_tests_folder(
    "configs/test_double_neuron_network_relay.json"
)


class TestTargetting(unittest.TestCase):
    def test_representatives(self):
        """
            Test that 1 cell per non-relay cell model is chosen.
        """
        config = JSONConfig(double_nn_config)
        scaffold = Scaffold(config)
        scaffold.compile_network()
        adapter = scaffold.create_adapter("neuron")
        targets = adapter.devices["test_representatives"].get_targets()
        self.assertEqual(
            1,
            len(targets),
            "Targetting type `representatives` did not return the correct amount of representatives.",
        )
