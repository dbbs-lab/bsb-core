import unittest, os, sys, numpy as np, h5py
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.morphologies import Morphology, Branch, _Labels
from bsb.storage import Storage
from bsb.exceptions import *
from test_setup import get_morphology, NumpyTestCase


class TestIO(NumpyTestCase, unittest.TestCase):
    def test_swc_2comp(self):
        m = Morphology.from_swc(get_morphology("2comp.swc"))
        self.assertEqual(2, len(m), "Expected 2 points on the morphology")
        self.assertEqual(1, len(m.roots), "Expected 1 root on the morphology")
        self.assertClose([1, 1], m.properties, "tags not loaded")
        self.assertClose([1, 1], m.tags, "tags should be all soma")
        self.assertClose(0, m.labels, "swc import is unlabelled")

    def test_swc_2root(self):
        m = Morphology.from_swc(get_morphology("2root.swc"))
        self.assertEqual(2, len(m), "Expected 2 points on the morphology")
        self.assertEqual(2, len(m.roots), "Expected 2 roots on the morphology")

    def test_swc_branch_filling(self):
        m = Morphology.from_swc(get_morphology("3branch.swc"))
        # SWC specifies child-parent edges, when translating that to branches, at branch
        # points some points need to be duplicated: there's 4 samples (SWC) and 2 child
        # branches -> 2 extra points == 6 points
        self.assertEqual(6, len(m), "Expected 6 points on the morphology")
        self.assertEqual(3, len(m.branches), "Expected 3 branches on the morphology")
        self.assertEqual(1, len(m.roots), "Expected 1 root on the morphology")

    def test_known(self):
        # TODO: Check the morphos visually with glover
        m = Morphology.from_swc(get_morphology("PurkinjeCell.swc"))
        self.assertEqual(3834, len(m), "Amount of point on purkinje changed")
        self.assertEqual(459, len(m.branches), "Amount of branches on purkinje changed")
        self.assertClose(
            42.45157433053635,
            np.mean(m.points),
            "value of the universe, life and everything changed.",
        )


class TestRepositories(unittest.TestCase):
    def test_empty_repository(self):
        pass

    def test_empty(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            ds = g.create_dataset("data", data=np.empty((0, 5)))
            ds.attrs["labels"] = json.dumps({0: []})
            ds.attrs["properties"] = []
            g.create_dataset("graph", data=[])
        mr = Storage("hdf5", "test.h5").morphologies
        m = mr.load("M")
        msg = "Empty morfo should not have root branches"
        self.assertEqual(0, len(m.roots), msg)
        msg = "Empty morfo should not have branches"
        self.assertEqual(0, len(m.branches), msg)
        msg = "Empty morfo should not have points"
        self.assertEqual(0, len(m.flatten()), msg)
        self.assertEqual(0, len(m), msg)
        self.assertTrue(m._check_shared(), "Empty morpho not shared")

    def test_empty_branches(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            ds = g.create_dataset("data", data=np.empty((0, 5)))
            ds.attrs["labels"] = json.dumps({0: []})
            ds.attrs["properties"] = []
            g.create_dataset("graph", data=[[0, -1], [0, -1], [0, -1]])
        mr = Storage("hdf5", "test.h5").morphologies
        m = mr.load("M")
        msg = "Empty unattached branches should still be root."
        self.assertEqual(3, len(m.roots), msg)
        self.assertEqual(3, len(m.branches), "Missing branch")
        msg = "Empty morfo should not have points, even when it has empty branches"
        self.assertEqual(0, len(m), msg)
        self.assertEqual(0, len(m.flatten()), msg)
        self.assertTrue(m._check_shared(), "Load should produce shared")

    def test_single_branch_single_element(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            ds = g.create_dataset("data", data=np.ones((1, 5)))
            ds.attrs["labels"] = json.dumps({1: []})
            ds.attrs["properties"] = []
            g.create_dataset("graph", data=[[0, -1]])
        mr = Storage("hdf5", "test.h5").morphologies
        m = mr.load("M")
        msg = "Single point unattached branches should still be root."
        self.assertEqual(1, len(m.roots), msg)
        self.assertEqual(1, len(m.branches), "Missing branch")
        msg = "Flatten of single point should produce 1 x 3 matrix."
        self.assertEqual((1, 3), m.flatten().shape, msg)
        msg = "should produce 1 element vector."
        self.assertEqual((1,), m.flatten_radii().shape, msg)
        self.assertEqual((1,), m.flatten_labels().shape, msg)
        msg = "Flatten without properties should produce n x 0 matrix."
        self.assertEqual((1, 0), m.flatten_properties().shape, msg)

    def test_multi_branch_single_element(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            data = np.ones((5, 5))
            data[:, 0] = np.arange(5) * 2
            data[:, 1] = np.arange(5) * 2
            data[:, 2] = np.arange(5) * 2
            data[:, 3] = np.arange(5) * 2
            ds = g.create_dataset("data", data=data)
            ds.attrs["labels"] = json.dumps({1: []})
            ds.attrs["properties"] = []
            g.create_dataset("graph", data=[[i + 1, -1] for i in range(5)])
            mr = Storage("hdf5", "test.h5").morphologies
            m = mr.load("M")
            msg = "Single point unattached branches should still be root."
            self.assertEqual(5, len(m.roots), msg)
            self.assertEqual(5, len(m.branches), "Missing branch")
            msg = "Flatten of single point branches should produce n-branch x n-vectors matrix."
            matrix = m.flatten()
            self.assertEqual((5, 3), matrix.shape, msg)
            msg = "Flatten produced an incorrect matrix"
            self.assertTrue(
                np.array_equal(np.array([[i * 2] * 3 for i in range(5)]), matrix), msg
            )

    def test_multi_branch_single_element_depth_first(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            data = np.ones((5, 5))
            data[:, 0] = np.arange(5) * 2
            data[:, 1] = np.arange(5) * 2
            data[:, 2] = np.arange(5) * 2
            data[:, 3] = np.arange(5) * 2
            ds = g.create_dataset("data", data=data)
            ds.attrs["labels"] = json.dumps({1: []})
            ds.attrs["properties"] = []
            g.create_dataset("graph", data=[[i + 1, -1] for i in range(4)] + [[5, 0]])
            mr = Storage("hdf5", "test.h5").morphologies
            m = mr.load("M")
            msg = "1 out of 5 branches was attached, 4 roots expected."
            self.assertEqual(4, len(m.roots), msg)
            self.assertEqual(5, len(m.branches), "Missing branch")

    def test_chain_empty_branches(self):
        pass

    def test_tree_empty_branches(self):
        pass

    def test_chain_branches(self):
        pass

    def test_chain_with_empty_branches(self):
        pass

    def test_tree_branches(self):
        pass

    def test_tree_with_empty_branches(self):
        pass


class TestMorphologies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def _branch(self, len):
        return Branch(np.ones((len, 3)), np.ones(len), _Labels.none(len), {})

    def test_branch_attachment(self):
        branch_A = self._branch(5)
        branch_B = self._branch(5)
        branch_C = self._branch(5)
        branch_D = self._branch(5)
        branch_A.attach_child(branch_B)
        branch_A.attach_child(branch_C)
        branch_B.attach_child(branch_D)
        self.assertEqual([branch_B, branch_C], branch_A._children)
        self.assertFalse(branch_A.is_terminal)
        self.assertFalse(branch_B.is_terminal)
        self.assertTrue(branch_C.is_terminal)
        self.assertTrue(branch_D.is_terminal)
        branch_A.detach_child(branch_C)
        self.assertIsNone(branch_C._parent)
        with self.assertRaises(ValueError):
            branch_A.detach_child(branch_D)
        self.assertEqual(branch_B, branch_D._parent)

    def test_properties(self):
        branch = Branch(
            np.array(
                [
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                ]
            ),
            np.array([0, 1, 2]),
        )
        self.assertEqual(3, branch.size, "Incorrect branch size")
        self.assertTrue(branch.is_terminal)
        branch.attach_child(branch)
        self.assertFalse(branch.is_terminal)


class TestMorphologyLabels(unittest.TestCase):
    def test_full_labels(self):
        v = len(Branch.vectors)
        branch = Branch(*(np.ones(v) for i in range(v)))
        branch.label_all("A", "B", "C")
        self.assertEqual(["A", "B", "C"], branch._full_labels)
        self.assertEqual(["A", "B", "C"], list(next(branch.label_walk())))
        self.assertTrue(all(["A", "B", "C"] == list(l) for l in branch.label_walk()))

    def test_point_labels(self):
        v = len(Branch.vectors)
        branch = Branch(*(np.ones(v) for i in range(v)))
        branch.label_points("A", [False, True] + [False] * (v - 2))
        self.assertEqual([], branch._full_labels)
        self.assertEqual(
            [[], ["A"]] + [[]] * (v - 2), list(map(list, branch.label_walk()))
        )

    def test_combo_labels(self):
        v = len(Branch.vectors)
        branch = Branch(*(np.ones(v) for i in range(v)))
        branch.label_points("A", [False, True] + [False] * (v - 2))
        branch.label_all("B")
        self.assertEqual(["B"], branch._full_labels)
        self.assertEqual(
            [["B"], ["B", "A"]] + [["B"]] * (v - 2), list(map(list, branch.label_walk()))
        )
