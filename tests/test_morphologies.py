import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.morphologies import Morphology, Branch
from bsb.storage.engines.hdf5 import morphology_repository as _mr_module
from bsb.exceptions import *
from scipy.spatial.transform import Rotation


@unittest.skip("Re-enabling tests gradually while advancing v4.0 rework")
class TestRepositories(unittest.TestCase):
    def test_empty_repository(self):
        pass

    def test_empty(self):
        with h5py.File("test.h5", "w") as f:
            g = f.create_group("morphologies")
            g = g.create_group("M")
            g.create_group("branches")
            m = _mr_module._morphology(g)
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
                m = _mr_module._morphology(g)
            b0.create_dataset("y", data=[])
            b0.create_dataset("z", data=[])
            b0.create_dataset("radii", data=[])
            m = _mr_module._morphology(g)
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
            b0.create_group("labels")
            b0.create_dataset("x", data=[1])
            b0.create_dataset("y", data=[1])
            b0.create_dataset("z", data=[1])
            b0.create_dataset("radii", data=[1])
            m = _mr_module._morphology(g)
            msg = "Single point unattached branches should still be root."
            self.assertEqual(1, len(m.roots), msg)
            self.assertEqual(1, len(m.branches), "Missing branch")
            msg = "Flatten of single point should produce 1 x n-vectors matrix."
            self.assertEqual((1, len(Branch.vectors)), m.flatten(matrix=True).shape, msg)
            t = m.flatten()
            msg = "Single empty branch morfo flatten to vector should produce vectors."
            self.assertEqual(len(Branch.vectors), len(t), msg)
            p = m.roots[0].points
            msg = "Single point branch.points should return (1, 4) ndarray."
            self.assertEqual((1, len(Branch.vectors)), p.shape, msg)
            msg = "Point data of branch.points incorrect."
            self.assertEqual([1, 1, 1, 1], list(p[0]), msg)
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

    def test_int_ordered_iter(self):
        unit = _mr_module._is_invalid_order
        unit2 = _mr_module._int_ordered_iter
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

    def test_branch_nargs(self):
        v = len(Branch.vectors)
        Branch(*(np.ones(i) for i in range(v)))
        with self.assertRaises(TypeError):
            Branch(*(np.ones(i) for i in range(v - 1)))
        with self.assertRaises(TypeError):
            Branch(*(np.ones(i) for i in range(v + 1)))

    def test_branch_attachment(self):
        v = len(Branch.vectors)
        branch_A = Branch(*(np.ones(v) for i in range(v)))
        branch_B = Branch(*(np.ones(v) for i in range(v)))
        branch_C = Branch(*(np.ones(v) for i in range(v)))
        branch_D = Branch(*(np.ones(v) for i in range(v)))
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

    def test_chaining(self):
        branch = Branch(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        m = Morphology([branch])
        r = Rotation.from_euler("z", 0)
        res = m.rotate(r).root_rotate(r).translate([0, 0, 0]).collapse().close_gaps()
        self.assertEqual(m, res, "chaining calls should return self")


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
