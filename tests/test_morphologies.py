import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bsb.output
from bsb.morphologies import Branch
from bsb.exceptions import *


class TestRepositories(unittest.TestCase):
    def test_empty_repository(self):
        pass


class TestMorphologies(unittest.TestCase):
    def test_int_ordered_iter(self):
        unit = bsb.output._is_invalid_order
        unit2 = bsb.output._int_ordered_iter
        # Check sequence
        self.assertFalse(unit([]))
        self.assertFalse(unit([0]))
        self.assertFalse(unit([0, 1]))
        self.assertFalse(unit([0, 1, 2]))
        self.assertFalse(unit([0, 1, 2, 3]))
        # Check start
        self.assertTrue(unit([1]))
        # Check gaps
        self.assertTrue(unit([0, 2]))
        self.assertTrue(unit([0, 1, 2, 4]))
        # Check empty str
        self.assertRaises(MorphologyDataError, unit2, {"": None})
        # Check str
        self.assertRaises(MorphologyDataError, unit2, {"a": None})
        # Check hex
        self.assertRaises(
            MorphologyDataError, unit2, {**{str(i): None for i in range(10)}, "a": None}
        )
        # Check neg
        self.assertRaises(MorphologyDataError, unit2, {"-1": None, "0": None})
        self.assertRaises(MorphologyDataError, unit2, {"-1": None, "0": None, "1": None})

    def test_empty(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            g.create_group("branches")
            m = bsb.output._morphology(None, g)
        msg = "Empty morfo should not have root branches"
        self.assertEqual(0, len(m.roots), msg)
        msg = "Empty morfo should not have branches"
        self.assertEqual(0, len(m.branches), msg)
        msg = "Empty morfo should not have points"
        self.assertEqual(0, len(m.flatten(matrix=True)), msg)
        t = m.flatten()
        msg = "Empty morfo flatten to vector should produce vectors."
        self.assertEqual(len(Branch.vectors), len(t), msg)
        for i, v in enumerate(Branch.vectors):
            with self.subTest(vector=v):
                msg = f"Flatten to vector produced non-empty {v}"
                self.assertEqual(0, len(t[i]), msg)
        self.assertEqual(0, len(m.to_compartments()), "Empty morfo should not have comps")

    def test_single_empty_branch(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            b = g.create_group("branches")
            b0 = b.create_group("0")
            b0.create_dataset("x", data=[])
            b0.create_group("labels")
            with self.assertRaises(MorphologyDataError):
                m = bsb.output._morphology(None, g)
            b0.create_dataset("y", data=[])
            b0.create_dataset("z", data=[])
            b0.create_dataset("radii", data=[])
            m = bsb.output._morphology(None, g)
            msg = "Empty unattached branches should still be root."
            self.assertEqual(1, len(m.roots), msg)
            self.assertEqual(1, len(m.branches), "Missing branch")
            msg = "Empty morfo should not have points, even when it has empty branches"
            self.assertEqual(0, len(m.flatten(matrix=True)), msg)
            t = m.flatten()
            msg = "Single empty branch morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            for i, v in enumerate(Branch.vectors):
                with self.subTest(vector=v):
                    msg = f"Flatten to vector produced non-empty {v}"
                    self.assertEqual(0, len(t[i]), msg)
            msg = "Empty branch morfo should not have comps"
            self.assertEqual(0, len(m.to_compartments()), msg)

    def test_single_branch_single_element(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            b = g.create_group("branches")
            b0 = b.create_group("0")
            b0.create_dataset("x", data=[1])
            b0.create_group("labels")
            b0.create_dataset("y", data=[1])
            b0.create_dataset("z", data=[1])
            b0.create_dataset("radii", data=[1])
            m = bsb.output._morphology(None, g)
            msg = "Single point unattached branches should still be root."
            self.assertEqual(1, len(m.roots), msg)
            self.assertEqual(1, len(m.branches), "Missing branch")
            msg = "Flatten of single point should produce 1 x n-vectors matrix."
            self.assertEqual((1, len(Branch.vectors)), m.flatten(matrix=True).shape, msg)
            t = m.flatten()
            msg = "Single empty branch morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            msg = "Single point branches should not produce comps"
            self.assertEqual(0, len(m.to_compartments()), msg)

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
            m = bsb.output._morphology(None, g)
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
            m = bsb.output._morphology(None, g)
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
            m = bsb.output._morphology(None, g)
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
