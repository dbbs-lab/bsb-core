import unittest, os, sys, numpy as np, h5py, importlib
from bsb.core import Scaffold
from bsb.exceptions import (
    MorphologyDataError,
    MorphologyError,
    MissingMorphologyError,
)
from bsb.postprocessing import SpoofDetails


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


_config = relative_to_tests_folder("configs/test_double_neuron.json")


def neuron_installed():
    return importlib.util.find_spec("neuron")


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
@unittest.skipIf(not neuron_installed(), "NEURON is not importable.")
class TestPostProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        test_setup.prep_morphologies()

    def test_spoofing(self):
        """
        Assert that fake detailed connections can be made
        """
        config = JSONConfig(file=_config)
        scaffold = Scaffold(config)
        scaffold.compile_network()
        original_connections = len(scaffold.cell_connections_by_tag["connection"])
        sd = SpoofDetails()
        sd.presynaptic = "from_cell"
        sd.postsynaptic = "to_cell"
        sd.scaffold = scaffold
        # Raise error because here's no morphologies registered for the cell types.
        with self.assertRaises(
            MorphologyDataError, msg="Missing morphologies during spoofing not caught."
        ):
            sd.after_connectivity()
        # Add some morphologies
        setattr(
            config.cell_types["from_cell"].morphology,
            "detailed_morphologies",
            {"names": ["GranuleCell"]},
        )
        setattr(
            config.cell_types["to_cell"].morphology,
            "detailed_morphologies",
            {"names": ["GranuleCell"]},
        )
        # Run the spoofing again
        sd.after_connectivity()
        cs = scaffold.get_connectivity_set("connection")
        scaffold.compile_output()
        # Use the intersection property. It throws an error should the detailed
        # information be missing
        try:
            i = cs.intersections
            for inter in i:
                fl = inter.from_compartment.labels
                tl = inter.to_compartment.labels
                self.assertIn("axon", fl, "From compartment type is not an axon")
                self.assertIn("dendrites", tl, "From compartment type is not a dendrite")
            self.assertNotEqual(len(i), 0, "Empty intersection data")
            self.assertEqual(
                len(i), original_connections, "Different amount of spoofed connections"
            )
        except MissingMorphologyError:
            self.fail("Could not find the intersection data on spoofed set")
        # Set both types to relays and try spoofing again
        config.cell_types["from_cell"].relay = True
        config.cell_types["to_cell"].relay = True
        with self.assertRaises(
            MorphologyError, msg="Did not catch double relay spoofing!"
        ):
            sd.after_connectivity()
