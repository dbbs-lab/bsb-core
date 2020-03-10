import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.core import Scaffold, from_hdf5
from scaffold.config import JSONConfig
from scaffold.models import PlacementSet, Cell
from scaffold.exceptions import DatasetNotFoundError


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


double_neuron_config = relative_to_tests_folder("configs/test_double_neuron.json")


class TestPlacementSets(unittest.TestCase):
    """
        Check if the scaffold can create a single cell type.
    """

    @classmethod
    def setUpClass(self):
        super(TestPlacementSets, self).setUpClass()
        config = JSONConfig(file=double_neuron_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()

    def test_hdf5_structure(self):
        with h5py.File(self.scaffold.output_formatter.file, "r") as h:
            for key in ["from", "to"]:
                group = h["cells/placement/" + key + "_cell"]
                self.assertTrue(
                    "identifiers" in group,
                    "Identifiers dataset missing for the " + key + "_cell",
                )
                self.assertTrue(
                    "positions" in group,
                    "Positions dataset missing for the " + key + "_cell",
                )
                self.assertEqual(
                    group["positions"].shape,
                    (4, 3),
                    "Incorrect position dataset size for the " + key + "_cell",
                )
                self.assertTrue(
                    group["positions"].dtype == np.float64,
                    "Incorrect position dataset dtype ({}) for the ".format(
                        group["positions"].dtype
                    )
                    + key
                    + "_cell",
                )
                self.assertEqual(
                    group["identifiers"].shape,
                    (2,),
                    "Incorrect or noncontinuous identifiers dataset size for the "
                    + key
                    + "_cell",
                )
                self.assertTrue(
                    group["identifiers"].dtype == np.int32,
                    "Incorrect identifiers dataset dtype ({}) for the ".format(
                        group["identifiers"].dtype
                    )
                    + key
                    + "_cell",
                )

    def test_placement_set_properties(self):
        for key in ["from", "to"]:
            cell_type = self.scaffold.get_cell_type(key + "_cell")
            ps = PlacementSet(self.scaffold.output_formatter, cell_type)
            self.assertEqual(
                ps.identifiers.shape,
                (4,),
                "Incorrect identifiers shape for " + key + "_cell",
            )
            self.assertEqual(
                ps.positions.shape,
                (4, 3),
                "Incorrect positions shape for " + key + "_cell",
            )
            self.assertRaises(DatasetNotFoundError, lambda: ps.rotations)
            self.assertEqual(
                type(ps.cells[0]), Cell, "PlacementSet.cells did not return Cells"
            )
            self.assertEqual(
                ps.cells[1].id,
                1 if key == "from" else 5,
                "PlacementSet.cells identifiers incorrect",
            )
            self.assertEqual(
                ps.cells[1].position.shape,
                (3,),
                "PlacementSet.cells positions wrong shape",
            )
