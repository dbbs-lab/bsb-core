import unittest, os, sys, numpy as np, h5py
import json
import itertools

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
        self.assertEqual(
            42.45157433053635,
            np.mean(m.points),
            "value of the universe, life and everything changed.",
        )
        m = Morphology.from_file(get_morphology("GolgiCell.asc"))
        self.assertEqual(5105, len(m), "Amount of point on purkinje changed")
        self.assertEqual(227, len(m.branches), "Amount of branches on purkinje changed")
        self.assertEqual(
            -11.14412080401295,
            np.mean(m.points),
            "something in the points changed.",
        )

    def test_shared_labels(self):
        m = Morphology.from_swc(get_morphology("PurkinjeCell.swc"))
        m2 = Morphology.from_swc(get_morphology("PurkinjeCell.swc"))
        l = m._shared._labels.labels
        self.assertIsNot(l, m2._shared._labels.label, "reload shares state")
        for b in m.branches:
            self.assertTrue(l is b._labels.labels, "Labels should be shared")
            l = b._labels.labels


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
        self.assertEqual({}, m.flatten_properties(), msg)

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


class TestMorphologies(NumpyTestCase, unittest.TestCase):
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

    def test_optimize(self):
        b1 = self._branch(3)
        b1.set_properties(smth=np.ones(len(b1)))
        b2 = self._branch(3)
        b2.label("oy")
        b2.translate([100, 100, 100])
        b2.set_properties(other=np.zeros(len(b2)), smth=np.ones(len(b2)))
        b3 = self._branch(3)
        b3.translate([200, 200, 200])
        b3.label("vey")
        b3.set_properties(other=np.ones(len(b3)))
        b4 = self._branch(3)
        b4.label("oy", "vey")
        b5 = self._branch(3)
        b5.label("oy")
        b5.translate([100, 100, 100])
        b6 = self._branch(3)
        b6.translate([200, 200, 200])
        b6.label("vey", "oy")
        m = Morphology([b1, b2, b3, b4, b5, b6])
        m.optimize()
        self.assertTrue(m._is_shared, "Should be shared after opt")
        self.assertEqual(18, len(m), "opt changed n points")
        self.assertClose(
            np.array([[1, 1, 1, 101, 101, 101, 201, 201, 201] * 2] * 3).T, m.points
        )
        # Since `hash`'s salt changes each run, the order in which the labels get sorted
        # can be different each run, but the order of occurence insensitive pattern
        # 0-1-2-3-1-3 stays the same.
        abcd = {}
        counter = itertools.count()
        for x in m._shared._labels:
            if x not in abcd:
                abcd[x] = next(counter)
        self.assertClose(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 3, 3, 3],
            np.vectorize(abcd.get)(m._shared._labels),
        )
        self.assertClose(1, m.smth[:6], "prop concat failed")
        self.assertNan(m.smth[6:], "prop concat failed")
        self.assertClose(0, m.other[3:6], "prop concat failed")
        self.assertClose(1, m.other[6:9], "prop concat failed")
        self.assertNan(m.other[9:], "prop concat failed")
        self.assertNan(m.other[:3], "prop concat failed")
        # Test DFS reorder of opt
        b1.attach_child(b3)
        m.roots.remove(b3)
        b4.attach_child(b6)
        m.roots.remove(b6)
        m.optimize(force=True)
        self.assertClose(
            np.array([[1, 1, 1, 201, 201, 201, 101, 101, 101] * 2] * 3).T, m.points
        )
        # Compare opt to flatten
        self.assertEqual(
            m.other[3:9].tolist(), m.flatten_properties()["other"][3:9].tolist()
        )
        l1 = m._shared._labels
        l2 = m.flatten_labels()
        self.assertClose(l1, l2, "opt v flatten labels discrepancy")
        self.assertEqual(l1.labels, l2.labels, "opt v flatten labels discrepancy")


class TestMorphologyLabels(NumpyTestCase, unittest.TestCase):
    def test_labels(self):
        a = _Labels.none(10)
        self.assertEqual({0: set()}, a.labels, "none labels should be empty")
        self.assertClose(0, a, "none labels should zero")
        a.label(["ello"], [1, 2])
        self.assertEqual({0: set(), 1: {"ello"}}, a.labels)
        self.assertClose([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], a)
        a.label(["ello", "goodbye"], [1, 2, 3, 4])
        self.assertEqual({0: set(), 1: {"ello"}, 2: {"ello", "goodbye"}}, a.labels)
        self.assertClose([0, 2, 2, 2, 2, 0, 0, 0, 0, 0], a)
        a.label(["goodbye"], [5, 6])
        self.assertEqual(
            {0: set(), 1: {"ello"}, 2: {"ello", "goodbye"}, 3: {"goodbye"}}, a.labels
        )
        self.assertClose([0, 2, 2, 2, 2, 3, 3, 0, 0, 0], a)
        a.label(["ello"], [9])
        self.assertEqual(
            {0: set(), 1: {"ello"}, 2: {"ello", "goodbye"}, 3: {"goodbye"}}, a.labels
        )
        self.assertClose([0, 2, 2, 2, 2, 3, 3, 0, 0, 1], a)
        a.label(["ello"], [*range(10)])
        self.assertClose([1, 2, 2, 2, 2, 2, 2, 1, 1, 1], a)
        a.label(["goodbye"], [*range(10)])
        self.assertClose([2] * 10, a)

    def test_branch_labels(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        a = b._labels
        self.assertEqual({0: set()}, a.labels, "none labels should be empty")
        self.assertClose(0, a, "none labels should zero")
        b.label("ello")
        self.assertClose(1, a, "full labelling failed")
        b.label("so long", "goodbye", "sayonara")
        self.assertClose(2, a, "multifull labelling failed")
        self.assertEqual(
            {0: set(), 1: {"ello"}, 2: {"ello", "so long", "goodbye", "sayonara"}},
            a.labels,
        )
        b.label([1, 3], "wow")
        self.assertClose([2, 3, 2, 3, 2, 2, 2, 2, 2, 2], a, "specific point label failed")

    def test_copy_labels(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.label("ello")
        b.label("so long", "goodbye", "sayonara")
        b.label([1, 3], "wow")
        b2 = b.copy()
        self.assertEqual(len(b), len(b2), "copy changed n points")
        self.assertEqual(b._labels.labels, b2._labels.labels, "copy changed labelset")
        self.assertIsNot(b._labels.labels, b2._labels.labels, "copy shares labels")

    def test_concat(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.label("ello")
        b2 = Branch([[0] * 3] * 10, [1] * 10)
        b2.label("not ello")
        # Both branches have a different definition for `1`, so concat should map them.
        self.assertClose(1, b._labels, "should all be labelled to 1")
        self.assertClose(1, b2._labels, "should all be labelled to 1")
        self.assertNotEqual(b._labels.labels, b2._labels.labels, "should have diff def")
        concat = _Labels.concatenate(b._labels, b2._labels)
        self.assertClose([1] * 10 + [2] * 10, concat)
        self.assertEqual({0: set(), 1: {"ello"}, 2: {"not ello"}}, concat.labels)

    def test_select(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.label("ello")
        b2 = Branch([[0] * 3] * 10, [1] * 10)
        b3 = Branch([[0] * 3] * 10, [1] * 10)
        b4 = Branch([[0] * 3] * 10, [1] * 10)
        b3.attach_child(b4)
        b3.label([1], "ello")
        self.assertTrue(b3.contains_label("ello"))
        m = Morphology([b, b2, b3])
        bs = m.select("ello").branches
        self.assertEqual([b, b3], m.select("ello").roots)
        self.assertEqual([b, b3, b4], m.select("ello").branches)
        self.assertEqual(len(b), len(b.get_points_labelled("ello")))
        self.assertEqual(1, len(b3.get_points_labelled("ello")))
