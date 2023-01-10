import importlib
import os
import unittest

from bsb.core import Scaffold


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


double_nn_config = relative_to_tests_folder(
    "configs/test_double_neuron_network_relay.json"
)


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(not neuron_installed(), "NEURON is not importable.")
class TestTargetting(unittest.TestCase):
    def test_representatives(self):
        """
        Test that 1 cell per non-relay cell model is chosen.
        """
        from patch import p

        config = JSONConfig(double_nn_config)
        scaffold = Scaffold(config)
        scaffold.compile_network()
        adapter = scaffold.create_adapter("neuron")
        adapter.h = p
        adapter.load_balance()
        device = adapter.devices["test_representatives"]
        device.initialise_targets()
        targets = adapter.devices["test_representatives"].get_targets()
        self.assertEqual(
            1,
            len(targets),
            "Targetting type `representatives` did not return the correct amount of representatives.",
        )
