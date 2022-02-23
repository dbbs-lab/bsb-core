import unittest, os, sys, numpy as np, h5py
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.morphologies import Morphology, Branch, _Labels
from bsb.storage import Storage
from bsb.exceptions import *


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
            b = g.create_group("branches")
            for i in range(5):
                b0 = b.create_group(str(i))
                b0.create_dataset("x", data=[i * 2])
                b0.create_group("labels")
                b0.create_dataset("y", data=[i * 2])
                b0.create_dataset("z", data=[i * 2])
                b0.create_dataset("radii", data=[i * 2])
            m = _mr_module._morphology(g)
            msg = "Single point unattached branches should still be root."
            self.assertEqual(5, len(m.roots), msg)
            self.assertEqual(5, len(m.branches), "Missing branch")
            msg = "Flatten of single point branches should produce n-branch x n-vectors matrix."
            matrix = m.flatten(matrix=True)
            self.assertEqual((5, len(Branch.vectors)), matrix.shape, msg)
            msg = "Flatten produced an incorrect matrix"
            self.assertTrue(
                np.array_equal(np.array([[i * 2] * 4 for i in range(5)]), matrix), msg
            )
            t = m.flatten()
            msg = "Single point branches morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            self.assertEqual(5, len(t[0]), msg)
            for vec, v in zip(t, Branch.vectors):
                with self.subTest(vector=v):
                    msg = f"Vector {v} did not correctly list the branch data points"
                    self.assertTrue(np.array_equal([0, 2, 4, 6, 8], vec), msg)
            msg = "Single point branches should not produce comps"
            self.assertEqual(0, len(m.to_compartments()), msg)

    def test_multi_branch_single_element_depth_first(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            b = g.create_group("branches")
            for i in range(5):
                b0 = b.create_group(str(i))
                b0.create_dataset("x", data=[i * 2])
                b0.create_group("labels")
                b0.create_dataset("y", data=[i * 2])
                b0.create_dataset("z", data=[i * 2])
                b0.create_dataset("radii", data=[i * 2])
            b["4"].attrs["parent"] = 0
            m = _mr_module._morphology(g)
            msg = "1 out of 5 branches was attached, 4 roots expected."
            self.assertEqual(4, len(m.roots), msg)
            self.assertEqual(5, len(m.branches), "Missing branch")
            msg = "Flatten of single point branches should produce n-branch x n-vectors matrix."
            matrix = m.flatten(matrix=True)
            self.assertEqual((5, len(Branch.vectors)), matrix.shape, msg)
            msg = "Flatten produced an incorrect matrix"
            v = len(Branch.vectors)
            # We expect the data of the first root first, then its child (8) then the
            # other roots in order, as a simplest demonstration of the depth first iter.
            ematrix = np.array([[0] * v, [8] * v, [2] * v, [4] * v, [6] * v])
            self.assertTrue(np.array_equal(ematrix, matrix), msg)
            t = m.flatten()
            msg = "Single point branches morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            self.assertEqual(5, len(t[0]), msg)
            for vec, v in zip(t, Branch.vectors):
                with self.subTest(vector=v):
                    msg = f"Vector {v} did not correctly list the branch data points"
                    self.assertTrue(np.array_equal([0, 8, 2, 4, 6], vec), msg)
            msg = "Single point branches should not produce comps"
            self.assertEqual(0, len(m.to_compartments()), msg)

    def test_multiple_empty_branches(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            b = g.create_group("branches")
            for i in range(5):
                b0 = b.create_group(str(i))
                b0.create_dataset("x", data=[])
                b0.create_dataset("y", data=[])
                b0.create_dataset("z", data=[])
                b0.create_dataset("radii", data=[])
                b0.create_group("labels")
            m = _mr_module._morphology(g)
            msg = "Empty unattached branches should still be root."
            self.assertEqual(5, len(m.roots), msg)
            msg = "Missing branch"
            self.assertEqual(5, len(m.branches), msg)
            msg = "Empty morfo should not have points, even when it has empty branches"
            self.assertEqual(0, len(m.flatten(matrix=True)), msg)
            t = m.flatten()
            msg = "Empty branches morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            for i, v in enumerate(Branch.vectors):
                with self.subTest(vector=v):
                    msg = f"Flatten to vector produced non-empty {v}"
                    self.assertEqual(0, len(t[i]), msg)
            msg = "Empty branches morfo should not have comps"
            self.assertEqual(0, len(m.to_compartments()), msg)
            for branch in m.branches:
                msg = "Unattached branches should not have parents"
                self.assertIsNone(branch._parent, msg)
                msg = "Setup without any attached branches should not produce branches with children"
                self.assertFalse(bool(branch._children), msg)

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

    def test_mr_labels(self):
        v = len(Branch.vectors)
        branch = Branch(*(np.ones(v) for i in range(v)))
        branch.label_points("A", [False, True] + [False] * (v - 2))
        branch.label_all("B")
        m = Morphology([branch])
        mr = _mr_module.MorphologyRepository("tmp.h5")
        mr.get_handle("w")
        mr.save("test", m)
        m_loaded = mr.load("test")
        branch_loaded = m_loaded.roots[0]
        self.assertEqual(["B"], branch_loaded._full_labels)
        self.assertEqual(
            [["B"], ["B", "A"]] + [["B"]] * (v - 2),
            list(map(list, branch_loaded.label_walk())),
        )


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
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        self.assertEqual(3, branch.size, "Incorrect branch size")
        self.assertTrue(branch.is_terminal)
        branch.attach_child(Branch(*(np.ones(0) for i in range(len(Branch.vectors)))))
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
