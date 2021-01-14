# Testing rotated_morphologies
import unittest, os, sys, numpy as np, h5py, test_setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig
from bsb.models import MorphologySet, PlacementSet
from bsb.output import MorphologyRepository, MorphologyCache
from bsb.morphologies import Morphology, Branch


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


config_file = relative_to_tests_folder(
    "configs/test_double_neuron_network_rotations.json"
)


class TestMorphologyCache(unittest.TestCase):
    """
    Test the creation of a morphology cache with rotated versions of some basic morphologies

    """

    @classmethod
    def setUpClass(self):
        import dbbs_models

        test_setup.prep_morphologies()
        test_setup.prep_rotations()

        super().setUpClass()
        config = JSONConfig(config_file)
        self.scaffold = Scaffold(config)
        mr = MorphologyRepository(test_setup.mr_rot_path)
        self.scaffold.morphology_repository = mr
        self.morphology_cache = MorphologyCache(mr)
        self.morphologies_start = ["GranuleCell", "GolgiCell", "GolgiCell_A"]
        self.morphologies_rotated = mr.list_morphologies(include_rotations=True)

    def test_morphology_repository(self):
        step = test_setup.rotations_step
        # Check if the rotated morphologies (at some significant angles) exist in the new morphology repository
        for m in self.morphologies_start:
            self.assertTrue(
                m + "__0_0" in self.morphologies_rotated, "Missing 0 0 rotated morphology"
            )
            self.assertTrue(
                m + "__" + str(step[0]) + "_" + str(step[1]) in self.morphologies_rotated,
                "Missing " + str(step[0]) + " " + str(step[1]) + " rotated morphology",
            )
            self.assertTrue(
                m + "__" + str(step[0] * 2) + "_" + str(step[1] * 2)
                in self.morphologies_rotated,
                "Missing "
                + str(step[0] * 2)
                + " "
                + str(step[1] * 2)
                + " rotated morphology",
            )
            self.assertTrue(
                m + "__360_360" in self.morphologies_rotated,
                "Missing 360 360 rotated morphology",
            )

    def test_equal_rotations(self):
        # Verify that for 0 0 rotations and 360 360 rotations, compartments are in the same position
        for m in self.morphologies_start:
            morpho_0 = self.scaffold.morphology_repository.get_morphology(m + "__0_0")
            morpho_360 = self.scaffold.morphology_repository.get_morphology(
                m + "__360_360"
            )
            comp_start_0 = ()
            comp_start_360 = ()
            comp_end_0 = ()
            comp_end_360 = ()
            for c in range(len(morpho_0.compartments)):
                comp_start_0 = comp_start_0 + tuple(
                    np.around(morpho_0.compartments[c].start, decimals=3)
                )
                comp_start_360 = comp_start_360 + tuple(
                    np.around(morpho_360.compartments[c].start, decimals=3)
                )
                comp_end_0 = comp_end_0 + tuple(
                    np.around(morpho_0.compartments[c].end, decimals=3)
                )
                comp_end_360 = comp_end_360 + tuple(
                    np.around(morpho_360.compartments[c].end, decimals=3)
                )

            self.assertEqual(
                comp_start_0,
                comp_start_360,
                "Different starting compartments for 0 and 360 rotation of morphology "
                + m,
            )
            self.assertEqual(
                comp_end_0,
                comp_end_360,
                "Different ending compartments for 0 and 360 rotation of morphology " + m,
            )

        # Verify 2 rotations with rotation matrix around y and z axis
        # Ry = []    step[1]
        # Rz = []    step[0]


class TestMorhologySetsRotations(unittest.TestCase):
    """
    Test scaffold with cells associated to a certain rotated morphology

    """

    @classmethod
    def setUpClass(self):
        import dbbs_models

        test_setup.prep_morphologies()
        test_setup.prep_rotations()

        super().setUpClass()
        config = JSONConfig(config_file)
        self.scaffold = Scaffold(config)
        self.scaffold.morphology_repository = MorphologyRepository(test_setup.mr_rot_path)

    def test_morphology_map(self):
        # Create and place a set of 10 Golgi cells and assign them to a morphology based on their rotation
        cell_type = self.scaffold.get_cell_type("golgi_cell")
        positions = np.random.rand(9, 3)
        # Construct rotation matrix for cell_type
        phi_values = np.linspace(0.0, 360.0, num=3)
        theta_values = np.linspace(0.0, 360.0, num=3)
        phi_values = np.repeat(
            phi_values, 3
        )  # np.random.choice(len(phi_values), len(positions))
        theta_values = np.repeat(
            theta_values, 3
        )  # np.random.choice(len(theta_values), len(positions))
        rotations = np.vstack((phi_values, theta_values)).T
        # Place cells and generate hdf5 output
        self.scaffold.place_cells(
            cell_type, cell_type.placement.layer_instance, positions, rotations
        )
        self.scaffold.compile_output()
        ps = PlacementSet(self.scaffold.output_formatter, cell_type)
        ms = MorphologySet(self.scaffold, cell_type, ps)
        self.assertEqual(
            len(rotations),
            len(ms._morphology_index),
            "Not all cells assigned to a morphology!",
        )
        random_sel = np.random.choice(len(ms._morphology_index))
        morpho_sel = ms._morphology_map[ms._morphology_index[random_sel]]
        self.assertTrue(
            morpho_sel.find(
                "__"
                + str(int(rotations[random_sel, 0]))
                + "_"
                + str(int(rotations[random_sel, 1]))
            )
            != -1,
            "Wrong morphology map!",
        )


class TestRotation(unittest.TestCase):
    """
    Test the validity of rotations
    """

    def test_rotate(self):
        root = Branch(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        m = Morphology([root])
        v0 = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        # Store pre rotation checks
        x0 = m.compartments[0].end.copy()
        # Rotate
        m.rotate(v0, v)
        s = m.compartments[0].start
        # Verify rotations
        self.assertEqualPoints(s, [0.0, 0.0, 0.0], "Rotation moved the origin!")
        self.assertEqualPoints(m.compartments[0].end, [0.0, 1.0, 0.0], v0=v0, v=v, x0=x0)

    def assertEqualPoints(self, x, p, msg=None, v0=None, v=None, x0=None):
        """
        Assert that point `x` is equal to point `p`
        """
        # Assert that they are of the same non-zero dimensionality
        self.assertNotEqual(len(x), 0, "Empty input point x given")
        self.assertEqual(len(x), len(p), "Different dimensions for input points x and p")
        # Store the variables that were passed to print below
        args = locals()
        # Delete some internal args that don't need to be printed
        del args["self"]
        del args["msg"]
        # Parse the assert message
        assert_msg = (
            msg
            if msg
            else (
                "Incorrect rotation"
                + ((" of " + repr(x0)) if x0 is not None else "")
                + " to "
                + repr(x)
                + ". Expected "
                + repr(p)
                + (
                    (" after rotation of v0 " + repr(v0) + " to v " + repr(v))
                    if v0 is not None
                    else ""
                )
            )
            # Append a list of the arguments that led to this assertion error
            + "\n\n  " + "\n  ".join(str(k) + " = " + str(v) for k, v in args.items())
        )
        for xi, pi in zip(x, p):
            # Assert that all points in x and p are almost equal
            self.assertAlmostEqual(xi, pi, msg=assert_msg)
