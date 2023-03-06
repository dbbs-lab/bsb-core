import unittest, os, sys, numpy as np, h5py
import json
import itertools

from bsb.services import MPI
from bsb.morphologies import Morphology, Branch, MorphologySet, RotationSet
from bsb._encoding import EncodedLabels
from bsb.storage import Storage
from bsb.storage.interfaces import StoredMorphology
from bsb.exceptions import *
from bsb.unittest import get_morphology_path, NumpyTestCase
from scipy.spatial.transform import Rotation


class TestIO(NumpyTestCase, unittest.TestCase):
    def test_swc_2comp(self):
        m = Morphology.from_swc(get_morphology_path("2comp.swc"))
        self.assertEqual(2, len(m), "Expected 2 points on the morphology")
        self.assertEqual(1, len(m.roots), "Expected 1 root on the morphology")
        self.assertClose([1, 1], m.tags, "tags should be all soma")
        self.assertClose(1, m.labels, "labels should be all soma")
        self.assertEqual({0: set(), 1: {"soma"}}, m.labels.labels, "incorrect labelsets")

    def test_swc_2root(self):
        m = Morphology.from_swc(get_morphology_path("2root.swc"))
        self.assertEqual(2, len(m), "Expected 2 points on the morphology")
        self.assertEqual(2, len(m.roots), "Expected 2 roots on the morphology")

    def test_swc_branch_filling(self):
        m = Morphology.from_swc(get_morphology_path("3branch.swc"))
        # SWC specifies child-parent edges, when translating that to branches, at branch
        # points some points need to be duplicated: there's 4 samples (SWC) and 2 child
        # branches -> 2 extra points == 6 points
        self.assertEqual(6, len(m), "Expected 6 points on the morphology")
        self.assertEqual(3, len(m.branches), "Expected 3 branches on the morphology")
        self.assertEqual(1, len(m.roots), "Expected 1 root on the morphology")

    def test_known(self):
        # TODO: Check the morphos visually with glover
        m = Morphology.from_swc(get_morphology_path("PurkinjeCell.swc"))
        self.assertEqual(3834, len(m), "Amount of point on purkinje changed")
        self.assertEqual(459, len(m.branches), "Amount of branches on purkinje changed")
        self.assertEqual(
            42.45157433053635,
            np.mean(m.points),
            "value of the universe, life and everything changed.",
        )
        m = Morphology.from_file(get_morphology_path("GolgiCell.asc"))
        self.assertEqual(5105, len(m), "Amount of point on purkinje changed")
        self.assertEqual(227, len(m.branches), "Amount of branches on purkinje changed")
        self.assertEqual(
            -11.14412080401295,
            np.mean(m.points),
            "something in the points changed.",
        )

    def test_shared_labels(self):
        m = Morphology.from_swc(get_morphology_path("PurkinjeCell.swc"))
        m2 = Morphology.from_swc(get_morphology_path("PurkinjeCell.swc"))
        lbl = m._shared._labels.labels
        self.assertIsNot(lbl, m2._shared._labels.label, "reload shares state")
        for b in m.branches:
            self.assertTrue(lbl is b._labels.labels, "Labels should be shared")
            lbl = b._labels.labels

    def test_graph_array(self):
        file = get_morphology_path("AA0048.swc")
        m = Morphology.from_swc(file)
        with open(str(file), "r") as f:
            content = f.read()
            data = np.array(
                [
                    swc_data
                    for line in content.split("\n")
                    if not line.strip().startswith("#")
                    and (swc_data := [float(x) for x in line.split() if x != ""])
                ]
            )
        converted_samples = m.to_graph_array()[:, 0].astype(int)
        converted_labels = m.to_graph_array()[:, 1].astype(int)
        converted_points = m.to_graph_array()[:, 2:5].astype(float)
        converted_radii = m.to_graph_array()[:, 5].astype(float)
        converted_parents = m.to_graph_array()[:, 6].astype(int)
        self.assertTrue(np.array_equal(data[:, 0].astype(int), converted_samples))
        self.assertTrue(np.array_equal(data[:, 1].astype(int), converted_labels))
        self.assertClose(data[:, 2:5].astype(float), converted_points)
        self.assertTrue(np.array_equal(data[:, 5].astype(float), converted_radii))
        self.assertTrue(np.array_equal(data[:, 6].astype(int), converted_parents))

        with self.assertRaises(NotImplementedError):
            b = _branch(10)
            b.label(["B", "A"], [0, 1, 2])
            m = Morphology([b])
            m.to_graph_array()


def _branch(len=3):
    return Branch(np.ones((len, 3)), np.ones(len), EncodedLabels.none(len), {})


class TestMorphologies(NumpyTestCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_branch_attachment(self):
        branch_A = _branch(5)
        branch_B = _branch(5)
        branch_C = _branch(5)
        branch_D = _branch(5)
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
        b1 = _branch(3)
        b1.set_properties(smth=np.ones(len(b1)))
        b2 = _branch(3)
        b2.label(["oy"])
        b2.translate([100, 100, 100])
        b2.set_properties(other=np.zeros(len(b2)), smth=np.ones(len(b2)))
        b3 = _branch(3)
        b3.translate([200, 200, 200])
        b3.label(["vey"])
        b3.set_properties(other=np.ones(len(b3)))
        b4 = _branch(3)
        b4.label(["oy", "vey"])
        b5 = _branch(3)
        b5.label(["oy"])
        b5.translate([100, 100, 100])
        b6 = _branch(3)
        b6.translate([200, 200, 200])
        b6.label(["vey", "oy"])
        m = Morphology([b1, b2, b3, b4, b5, b6])
        m.optimize()
        self.assertTrue(m._is_shared, "Should be shared after opt")
        self.assertEqual(18, len(m), "opt changed n points")
        self.assertClose(
            np.array([[1, 1, 1, 101, 101, 101, 201, 201, 201] * 2] * 3).T, m.points
        )
        self.assertClose(
            [0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            m._shared._labels,
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

    def test_chaining(self):
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
        m = Morphology([branch])
        r = Rotation.from_euler("z", 0)
        res = m.rotate(r).root_rotate(r).translate([0, 0, 0]).collapse().close_gaps()
        self.assertEqual(m, res, "chaining calls should return self")

    def test_simplification(self):
        def branch_one():
            return Branch(
                np.array([[0, 0, 0], [1, 1, 0], [0, 4, 0], [0, 6, 0], [2, 4, 8]]),
                np.array([0, 1, 2, 2, 1]),
            )

        def branch_two():
            return Branch(
                np.empty((0, 3)),
                np.array([]),
            )

        m = Morphology([branch_one()])
        m.simplify(epsilon=10)
        self.assertClose(
            m.branches[0].points,
            np.array([[0, 0, 0], [2, 4, 8]]),
            "It has failed base rdp",
        )

        m_empty = Morphology([branch_two()])
        m_empty.simplify(epsilon=1)
        self.assertClose(
            m_empty.branches[0].points,
            np.empty((0, 3)),
            "It has failed rdp on empty branch",
        )

        b1 = branch_one()
        b1.attach_child(branch_two())
        m_chained = Morphology([b1])
        m_chained.simplify(epsilon=10)
        self.assertClose(
            m_chained.branches[0].points,
            np.array([[0, 0, 0], [2, 4, 8]]),
            "It has failed rdp on concatenated branches",
        )
        self.assertClose(
            m_chained.branches[1].points,
            np.empty((0, 3)),
            "It has failed rdp on concatenated branches",
        )

        # test epsilon values
        m_epsilon_0 = Morphology([branch_one()])
        m_epsilon_0.simplify(epsilon=0)
        self.assertClose(
            m_epsilon_0.branches[0].points,
            np.array([[0, 0, 0], [1, 1, 0], [0, 4, 0], [0, 6, 0], [2, 4, 8]]),
            "It has failed rdp with epsilon 0",
        )
        with self.assertRaises(ValueError, msg="It should throw a ValueError") as context:
            m_epsilon_0.simplify(epsilon=-1)

    def test_adjacency(self):
        target = {0: [1], 1: [2, 5], 2: [3, 4], 3: [], 4: [], 5: []}
        root = _branch(1)
        branch_A = _branch(5)
        branch_B = _branch(5)
        branch_C = _branch(5)
        branch_D = _branch(5)
        branch_E = _branch(5)
        branch_A.attach_child(branch_B)
        branch_B.attach_child(branch_C)
        branch_B.attach_child(branch_D)
        branch_A.attach_child(branch_E)
        root.attach_child(branch_A)
        m = Morphology([root])

        self.assertEqual(m.adjacency_dictionary, target)


class TestMorphologyLabels(NumpyTestCase, unittest.TestCase):
    def test_labels(self):
        a = EncodedLabels.none(10)
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
        b.label(["ello"])
        self.assertClose(1, a, "full labelling failed")
        b.label(["so long", "goodbye", "sayonara"])
        self.assertClose(2, a, "multifull labelling failed")
        self.assertEqual(
            {0: set(), 1: {"ello"}, 2: {"ello", "so long", "goodbye", "sayonara"}},
            a.labels,
        )
        b.label("wow", [1, 3])
        self.assertClose([2, 3, 2, 3, 2, 2, 2, 2, 2, 2], a, "specific point label failed")

    def test_copy_labels(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.label(["ello"])
        b.label(["so long", "goodbye", "sayonara"])
        b.label(["wow"], [1, 3])
        b2 = b.copy()
        self.assertEqual(len(b), len(b2), "copy changed n points")
        self.assertEqual(b._labels.labels, b2._labels.labels, "copy changed labelset")
        self.assertIsNot(b._labels.labels, b2._labels.labels, "copy shares labels")

    def test_concat(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.label(["ello"])
        b2 = Branch([[0] * 3] * 10, [1] * 10)
        b2.label(["not ello"])
        # Both branches have a different definition for `1`, so concat should map them.
        self.assertClose(1, b._labels, "should all be labelled to 1")
        self.assertClose(1, b2._labels, "should all be labelled to 1")
        self.assertNotEqual(b._labels.labels, b2._labels.labels, "should have diff def")
        concat = EncodedLabels.concatenate(b._labels, b2._labels)
        self.assertClose([1] * 10 + [2] * 10, concat)
        self.assertEqual({0: set(), 1: {"ello"}, 2: {"not ello"}}, concat.labels)

    def test_select(self):
        b = Branch([[0] * 3] * 10, [1] * 10)
        b.name = "B1"
        b.label(["ello"])
        b2 = Branch([[0] * 3] * 10, [1] * 10)
        b2.name = "B2"
        b3 = Branch([[0] * 3] * 10, [1] * 10)
        b3.name = "B3"
        b4 = Branch([[0] * 3] * 10, [1] * 10)
        b4.name = "B4"
        b3.attach_child(b4)
        b3.label(["ello"], [1])
        self.assertTrue(b3.contains_labels(["ello"]))
        m = Morphology([b, b2, b3])
        bs = m.subtree(["ello"]).branches
        self.assertEqual([b, b3, b4], m.subtree(["ello"]).branches)
        self.assertEqual(len(b), len(b.get_points_labelled(["ello"])))
        self.assertEqual(1, len(b3.get_points_labelled(["ello"])))

    def test_list_labels(self):
        b = _branch(10)
        c = _branch(10)
        b.attach_child(c)
        b.label(["B", "A"], [0, 1, 2])
        c.label(["B", "C", "D", "A"], [0, 1, 2, 5])
        m = Morphology([b])
        self.assertEqual(
            {0: [], 1: ["B", "A"], 2: ["B", "C", "D", "A"]},
            m.labelsets,
            "expected no and double labelset",
        )
        self.assertEqual(
            ["A", "B", "C", "D"], m.list_labels(), "expected sorted list of labels"
        )
        maskA = m.get_label_mask(["A"])
        self.assertEqual(7, np.sum(maskA), "expected 3 hits for A")
        self.assertEqual(["A", "B"], b.list_labels(), "expected sorted branch labels")
        self.assertEqual(
            ["A", "B", "C", "D"], c.list_labels(), "expected sorted branch labels"
        )

    def test_mlabel(self):
        b = _branch(10)
        m = Morphology([b])
        m.optimize()
        self.assertEqual(0, np.sum(m.get_label_mask(["A"])[:3]), "expected first 0 lbled")
        m.label(["B", "A"], [0, 1, 2])
        self.assertEqual(3, np.sum(m.get_label_mask(["A"])[:3]), "then first 3 lbled")


class TestPointSetters(NumpyTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.m = Morphology(
            [
                Branch(
                    [[i] * 3] * i,
                    [1] * i,
                    children=[Branch([[i] * 3] * i, [1] * i) for _ in range(i + 1)],
                )
                for i in range(5)
            ]
        )

    def test_branch_same_points(self):
        p = np.array(self.m.branches[5].points)
        self.m.branches[5].points = p
        self.assertClose(p, self.m.branches[5].points, "same points, so should be close")
        p = np.array(self.m.branches[5].radii)
        self.m.branches[5].radii = p
        self.assertClose(p, self.m.branches[5].radii, "same points, so should be close")

    def test_branch_doubleval_points(self):
        p = np.array(self.m.branches[5].points)
        self.m.branches[5].points *= 2
        self.assertClose(p * 2, self.m.branches[5].points, "expected doubled")
        p = np.array(self.m.branches[5].radii)
        self.m.branches[5].radii *= 2
        self.assertClose(p * 2, self.m.branches[5].radii, "expected doubled")

    def test_branch_doublenum_points(self):
        p = np.array(self.m.branches[5].points)
        self.m.branches[5].points = np.tile(p, (2, 1))
        self.assertEqual(len(p) * 2, len(self.m.branches[5].points), "expected doublenum")
        p = np.array(self.m.branches[5].radii)
        self.m.branches[5].radii = np.tile(p, (2, 1))
        self.assertEqual(len(p) * 2, len(self.m.branches[5].radii), "expected doublenum")

    def test_branch_invalid_points(self):
        with self.assertRaises(ValueError):
            # Numpy raises a clear ValueError
            self.m.branches[5].points = [[1], [1, 2]]
        with self.assertRaises(ValueError):
            # Numpy raises a clear ValueError
            self.m.branches[5].points = [[1, 2], [1, 2]]

    def test_points(self):
        with self.assertRaises(ValueError):
            # Unoptimized morpho, branches should raise ValueError when given empty arr
            self.m.points = []

    def test_reassign_points(self):
        len_pre = len(self.m.points)
        self.m.points = np.array(self.m.points)
        self.assertEqual(len_pre, len(self.m.points), "should've stayed same len")
        self.m.optimize()
        with self.assertRaises(ValueError):
            # Numpy raises ValueError because data doesn't fit
            self.m.points = []
        # Test that it remains functional after error state
        self.m.points = np.array(self.m.points)

    def test_empty_radii(self):
        self.m.radii = []
        self.assertEqual(0, len(self.m.radii), "should've erased radii")

    def test_reassign_radii(self):
        len_pre = len(self.m.radii)
        self.m.radii = np.array(self.m.radii)
        self.assertEqual(len_pre, len(self.m.radii), "should've stayed same len")

    def test_optimized(self):
        self.m.optimize()
        with self.assertRaises(ValueError):
            # Numpy raises ValueError because data doesn't fit
            self.m.radii = []


class TestMorphologySet(NumpyTestCase, unittest.TestCase):
    def _fake_loader(self, name):
        return StoredMorphology(name, lambda: Morphology([Branch([], [])]), dict())

    def _label_loader(self, name):
        def m():
            mo = Morphology(
                [Branch([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], [1] * 4)]
            )
            mo.label(["A"], [0, 1])
            mo.label(["B"], [1, 2])
            mo.label(["C"], [2, 3])
            return mo

        return StoredMorphology(name, m, dict())

    def setUp(self):
        self.sets = [
            MorphologySet([], []),
            MorphologySet([self._fake_loader("ello")], [0, 0, 0]),
        ]

    def test_names(self):
        self.assertEqual([], self.sets[0].names, "expected empty names list")
        self.assertEqual(["ello"], self.sets[1].names, "expected matching names list")

    def test_count(self):
        self.assertEqual(0, self.sets[0].count_morphologies(), "expected no morphologies")
        self.assertEqual(1, self.sets[1].count_morphologies(), "expected fake loader")

    def test_count_unique(self):
        self.assertEqual(0, self.sets[0].count_unique(), "expected no unique")
        self.assertEqual(1, self.sets[1].count_unique(), "expected fake loader")
        ms = MorphologySet([self._fake_loader("ello")] * 3, [0, 0, 0])
        self.assertEqual(1, ms.count_unique(), "expected fake loader")
        ms = MorphologySet(
            [
                self._fake_loader("ello"),
                self._fake_loader("ello"),
                StoredMorphology(
                    "oh", lambda: Morphology([Branch([[0, 0, 0]], [0])]), dict()
                ),
            ],
            [0, 0, 0],
        )
        self.assertEqual(2, ms.count_unique(), "expected 2 uniques")

    def test_oob(self):
        with self.assertRaises(IndexError):
            MorphologySet([self._fake_loader("ello")], [0, 1, 0])

    def test_hard_cache(self):
        cached = self.sets[1].iter_morphologies(hard_cache=True)
        d = None
        self.assertTrue(
            all((d := c if d is None else d) is d for c in cached),
            "hard cache should return identical objects",
        )
        uncached = self.sets[1].iter_morphologies()
        d = None
        self.assertTrue(
            all((d := c if d is None else d) is d for c in list(cached)[1:]),
            "soft cache should not be ident",
        )

    def test_unique(self):
        self.assertEqual(
            1,
            len([*self.sets[1].iter_morphologies(unique=True)]),
            "only 1 morph in unique set",
        )
        self.assertEqual(
            1, len([*self.sets[1].iter_meta(unique=True)]), "only 1 morph in unique set"
        )

    def test_filtered_get(self):
        ms = MorphologySet([self._label_loader("ello")], [0, 0, 0])
        ms.set_label_filter(["A"])
        m = ms.get(0, cache=False)
        self.assertEqual(2, len(m), "expected filtered morpho")
        self.assertClose([[0, 0, 0], [1, 1, 1]], m.points, "expected A labelled points")
        ms.set_label_filter(["B"])
        m = ms.get(0, cache=False)
        self.assertEqual(2, len(m), "expected filtered morpho")
        self.assertClose([[1, 1, 1], [2, 2, 2]], m.points, "expected B labelled points")

    def test_softcache_filtered_get(self):
        ms = MorphologySet([self._label_loader("ello")], [0, 0, 0])
        ms.set_label_filter(["A"])
        m = ms.get(0, cache=True)
        self.assertEqual(2, len(m), "expected filtered morpho")
        self.assertClose([[0, 0, 0], [1, 1, 1]], m.points, "expected A labelled points")
        ms.set_label_filter(["B"])
        m = ms.get(0, cache=True)
        self.assertEqual(2, len(m), "expected filtered morpho")
        self.assertClose([[1, 1, 1], [2, 2, 2]], m.points, "expected B labelled points")

    def test_hardcache_filtered_get(self):
        ms = MorphologySet([self._label_loader("ello")], [0, 0, 0])
        ms.set_label_filter(["A"])
        m1 = ms.get(0, hard_cache=True)
        self.assertEqual(2, len(m1), "expected filtered morpho")
        self.assertClose([[0, 0, 0], [1, 1, 1]], m1.points, "expected A labelled points")
        m2 = ms.get(0, hard_cache=True)
        self.assertEqual(2, len(m2), "expected filtered morpho")
        self.assertClose([[0, 0, 0], [1, 1, 1]], m2.points, "expected A labelled points")
        self.assertEqual(m1, m2, "expected identical morphos")
        ms.set_label_filter(["B"])
        m = ms.get(0, hard_cache=True)
        self.assertNotEqual(m1, m, "expected invalidated hard cache")
        self.assertEqual(2, len(m), "expected filtered morpho")
        self.assertClose([[1, 1, 1], [2, 2, 2]], m.points, "expected B labelled points")


class TestMorphometry(NumpyTestCase, unittest.TestCase):
    def setUp(self):
        # Toy branches
        self.b0 = Branch([], [])
        self.bzero1 = Branch([[0] * 3], [1])
        self.bzero_r1 = Branch([[1] * 3], [0])
        self.b1 = Branch([[1] * 3], [1])
        self.bzero2 = Branch([[0] * 3] * 2, [1] * 2)
        self.bzero_r2 = Branch([[0] * 3] * 2, [0] * 2)
        self.b2 = Branch([[1] * 3, [2] * 3], [1] * 2)
        self.bzero10 = Branch([[0] * 3] * 10, [1] * 10)
        self.bzero_r10 = Branch([[1] * 3] * 10, [0] * 10)
        self.b3 = Branch([[0, 0, 0], [3, 6 * np.sin(np.pi / 3), 0], [6, 0, 0]], [1] * 3)
        # Meaningful toy morphology
        m = Morphology.from_swc(get_morphology_path("test_morphometry.swc"))
        self.adjacency = m.branch_adjacency
        self.branches = m.branches

    def test_short_branch(self):
        for attr in (
            "euclidean_dist",
            "vector",
            "versor",
            "start",
            "max_displacement",
        ):
            with self.subTest(attr=attr):
                with self.assertRaises(EmptyBranchError):
                    getattr(self.b0, attr)

        for attr in (
            "versor",
            "max_displacement",
        ):
            with self.subTest(attr=attr):
                with self.assertRaises(EmptyBranchError):
                    getattr(self.bzero1, attr)

    def test_zero_len(self):
        for attr in ("euclidean_dist", "path_dist"):
            with self.subTest(attr=attr):
                self.assertEqual(getattr(self.b1, attr), 0)
                self.assertEqual(getattr(self.bzero1, attr), 0)
                self.assertEqual(getattr(self.bzero_r1, attr), 0)
                self.assertEqual(getattr(self.bzero2, attr), 0)
                self.assertEqual(getattr(self.bzero_r2, attr), 0)
                self.assertEqual(getattr(self.bzero10, attr), 0)
                self.assertEqual(getattr(self.bzero_r10, attr), 0)

    def test_known_len(self):
        self.assertClose(self.b3.path_dist, 12)
        self.assertClose(self.b3.euclidean_dist, 6)

    def test_adjacency(self):
        known_adj = {0: [1, 2], 1: [], 2: [3, 4, 5], 3: [], 4: [], 5: []}
        self.assertEqual(len(self.branches[0].children), 2)
        self.assertEqual(len(self.branches[2].children), 3)
        self.assertDictEqual(known_adj, self.adjacency)

    def test_start_end(self):
        self.assertClose(self.branches[0].start, [0.0, 1.0, 0.0])
        self.assertClose(self.branches[0].end, [0.0, 1.0, 0.0])
        self.assertClose(self.branches[1].start, [0.0, 1.0, 0.0])
        self.assertClose(self.branches[1].end, [-5.0, np.exp(5), 0.0])
        self.assertClose(self.branches[2].start, [0.0, 1.0, 0.0])
        self.assertClose(self.branches[2].end, [0.0, 11.0, 0.0])
        self.assertClose(self.branches[3].start, [0.0, 11.0, 0.0])
        self.assertClose(
            self.branches[3].end,
            [0.0 + 10 * np.cos(np.pi / 2), 11.0 + 10 * np.sin(np.pi / 2), 0.0],
        )
        self.assertClose(self.branches[4].start, [0.0, 11.0, 0.0])
        self.assertClose(
            self.branches[4].end,
            [0.0 + 10 * np.cos(np.pi / 3), 11.0 + 10 * np.sin(np.pi / 3), 0.0],
        )
        self.assertClose(self.branches[5].start, [0.0, 11.0, 0.0])
        self.assertClose(
            self.branches[5].end,
            [
                0.0 + 10 * np.cos((2 / 3) * np.pi),
                11.0 + 10 * np.sin((2 / 3) * np.pi),
                0.0,
            ],
        )

    def test_vectors(self):
        self.assertClose(self.branches[2].versor, [0.0, 1.0, 0.0])
        self.assertClose(self.branches[2].vector, [0.0, 10.0, 0.0])
        self.assertClose(self.branches[3].versor, [0, 1.0, 0.0])
        self.assertClose(self.branches[3].vector, [0, 10.0, 0.0])
        self.assertClose(
            self.branches[4].versor, [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0]
        )
        self.assertClose(
            self.branches[5].versor,
            [np.cos((2 / 3) * np.pi), np.sin((2 / 3) * np.pi), 0.0],
        )
        pass

    def test_displacement(self):
        self.assertClose(self.branches[2].max_displacement, 5.0)
        for b in self.branches[3:]:
            self.assertClose(b.max_displacement, 0, atol=1e-06)

    def test_fractal_dim(self):
        for b in self.branches[3:]:
            self.assertClose(b.fractal_dim, 1.0)


class TestSwcFiles(NumpyTestCase, unittest.TestCase):
    # Helper functions to create a toy morphology
    def generate_semicircle(self, center_x, center_y, radius, stepsize=0.01):
        x = np.arange(center_x, center_x + radius + stepsize, stepsize)
        y = np.sqrt(radius**2 - x**2)

        x = np.concatenate([x, x[::-1]])
        y = np.concatenate([y, -y[::-1]])
        z = np.zeros(y.shape)

        return x, y + center_y, z

    def generate_exponential(self, center_x, center_y, len=10, stepsize=0.1):
        x = np.arange(center_x, center_x + len + stepsize, stepsize)
        y = np.exp(x)
        z = np.zeros(y.shape)

        return -x, y + center_y, z

    def generate_radius(
        self, origin_x, origin_y, len=10, angle=(np.pi / 2), stepsize=0.1
    ):
        l = np.arange(0, len + stepsize, stepsize)
        x = l * np.cos(angle) + origin_x
        y = l * np.sin(angle) + origin_y
        z = np.zeros(y.shape)

        return x, y, z

    def setUp(self):
        # Creating the branches
        x_s, y_s, z_s = self.generate_semicircle(0, 6, 5, 0.01)
        x_e, y_e, z_e = self.generate_exponential(0, 0, 5, 0.01)
        x_ri, y_ri, z_ri = self.generate_radius(0, 11, len=10)
        x_rii, y_rii, z_rii = self.generate_radius(0, 11, angle=np.pi / 3, len=10)
        x_riii, y_riii, z_riii = self.generate_radius(
            0, 11, angle=(2 / 3) * np.pi, len=10
        )

        root = Branch(np.array([0.0, 1.0, 0.0]).reshape(1, 3), radii=1)
        exp_child = Branch(np.vstack((x_e, y_e, z_e)).T, radii=[1] * len(x_e))
        semi_child = Branch(np.vstack((x_s, y_s[::-1], z_s)).T, radii=[1] * len(x_s))
        ri_child = Branch(np.vstack((x_ri, y_ri, z_ri)).T, radii=[1] * len(x_ri))
        rii_child = Branch(np.vstack((x_rii, y_rii, z_rii)).T, radii=[1] * len(x_rii))
        riii_child = Branch(
            np.vstack((x_riii, y_riii, z_riii)).T, radii=[1] * len(x_riii)
        )
        semi_child.attach_child(ri_child)
        semi_child.attach_child(rii_child)
        semi_child.attach_child(riii_child)
        root.attach_child(exp_child)
        root.attach_child(semi_child)

        self.m = Morphology([root])

    def test_identity(self):
        m = Morphology.from_swc(get_morphology_path("test_morphometry.swc"))
        self.assertClose(m.points, self.m.points)


class TestBranchInsertion(NumpyTestCase, unittest.TestCase):
    def setUp(self):
        root = Branch(np.array([0.0, 0.0, 0.0]).reshape(1, 3), radii=1)
        x1 = np.arange(4.0, dtype=float)
        y1, z = np.zeros(len(x1), dtype=float), np.zeros(len(x1), dtype=float)
        b1 = Branch(((np.vstack((x1, y1, z)).T)).reshape(len(x1), 3), radii=[1] * len(x1))
        x2 = np.ones(len(x1), dtype=float) * 2
        y2 = np.arange(4.0, dtype=float)
        self.b2 = Branch(
            ((np.vstack((x2, y2, z)).T)).reshape(len(x1), 3), radii=[1] * len(x1)
        )
        root.attach_child(b1)
        self.m = Morphology([root])

    def test_insertion_points(self):
        insertion_pt = np.array([2.0, 0.0, 0.0])
        self.m.branches[1].insert_branch(self.b2, insertion_pt)
        # self.m.close_gaps()
        for c in self.m.branches[1].children:
            self.assertClose(insertion_pt, c.start)
        self.assertClose(self.m.branches[1].end, insertion_pt)
        for c in self.m.branches[1].children:
            self.assertClose(insertion_pt, c.start)

    def test_insertion_indices(self):
        target = {0: [1, 2], 1: [], 2: []}
        b = self.m.branches[1]
        self.assertRaises(IndexError, b.insert_branch, self.b2, -5)
        self.assertRaises(IndexError, b.insert_branch, self.b2, -1)
        self.m.branches[1].insert_branch(self.b2, 0)
        self.assertEqual(self.m.adjacency_dictionary, target)
        self.assertRaises(IndexError, b.insert_branch, self.b2, len(b))
        self.assertRaises(IndexError, b.insert_branch, self.b2, len(b) + 1)

    def test_hierarchy(self):
        target = {
            0: [1],
            1: [2, 4],
            2: [3],
            3: [],
            4: [5, 6],
            5: [],
            6: [7, 8, 9],
            7: [],
            8: [],
            9: [],
        }
        x0 = np.arange(4.0, dtype=float) + 3.0
        y0, z = np.zeros(len(x0), dtype=float), np.zeros(len(x0), dtype=float)
        b0 = Branch(((np.vstack((x0, y0, z)).T)).reshape(len(x0), 3), radii=[1] * len(x0))
        first_insertion_pt = np.array([2.0, 0.0, 0.0])
        self.m.branches[1].insert_branch(self.b2, first_insertion_pt)
        x = np.arange(start=3.0, stop=6.0, dtype=float)
        y3 = np.arange(3.0, dtype=float)
        y4 = -np.arange(3.0, dtype=float)
        z = np.zeros(len(x), dtype=float)
        b3 = Branch(((np.vstack((x, y3, z)).T)).reshape(len(x), 3), radii=[1] * len(x))
        b4 = Branch(((np.vstack((x, y4, z)).T)).reshape(len(x), 3), radii=[1] * len(x))
        second_insertion_pt = np.array([3.0, 0.0, 0.0])
        self.m.branches[3].insert_branch(b3, second_insertion_pt)
        self.m.branches[3].insert_branch(b4, second_insertion_pt)
        third_insertion_pt = np.array([1.0, 0.0, 0.0])
        y5 = -np.arange(3.0, dtype=float)
        x5 = np.ones(len(y5), dtype=float)
        z5 = np.zeros(len(y5), dtype=float)
        x6 = -np.arange(3.0, dtype=float)
        y6 = np.ones(len(x6), dtype=float) * 2
        z6 = np.zeros(len(x6), dtype=float)
        b5 = Branch(
            ((np.vstack((x5, y5, z5)).T)).reshape(len(x5), 3), radii=[1] * len(x5)
        )
        b6 = Branch(
            ((np.vstack((x6, y6, z6)).T)).reshape(len(x6), 3), radii=[1] * len(x6)
        )
        b5.attach_child(b6)
        b4.insert_branch(b0, third_insertion_pt)
        self.m.branches[1].insert_branch(b5, third_insertion_pt)
        self.m.close_gaps()
        for c in self.m.branches[3].children:
            self.assertClose(second_insertion_pt, c.start)
        self.assertEqual(self.m.adjacency_dictionary, target)


class TestMorphologyFiltering(NumpyTestCase, unittest.TestCase):
    def test_filter_none(self):
        m = Morphology([Branch(np.ones((5, 3)), np.ones(5))])
        m.label(["test_all"])
        m2 = m.as_filtered()
        self.assertIsNot(m, m2, "filtering without labels should return copy")
        self.assertEqual(len(m), len(m2), "filtering without labels should return all")

    def test_filter_all(self):
        m = Morphology([Branch(np.ones((5, 3)), np.ones(5))])
        m.label(["test_all"])
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(len(m), len(m2), "filtering all should return all")
        self.assertEqual(len(m.branches), len(m2.branches), "n branches change")
        self.assertEqual(1, len(m2.branches), "just 1 branch")

    def test_filter_split(self):
        m = Morphology([Branch(np.ones((5, 3)), np.ones(5))])
        split_one = np.ones(5, dtype=bool)
        split_one[2] = 0
        m.label(["test_all"], split_one)
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(4, len(m2), "filtering should return 4 filtered points")
        self.assertEqual(2, len(m2.branches), "expected split in the middle")
        self.assertEqual(2, len(m2.branches[0]), "expected 2 point branch")
        self.assertEqual(2, len(m2.branches[1]), "expected 2 point branch")

    def test_filter_trim_start(self):
        m = Morphology([Branch(np.ones((5, 3)), np.ones(5))])
        split_one = np.ones(5, dtype=bool)
        split_one[0] = 0
        m.label(["test_all"], split_one)
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(4, len(m2), "filtering should return 4 filtered points")
        self.assertEqual(1, len(m2.branches), "expected trim of the start")
        self.assertEqual(4, len(m2.branches[0]), "expected 4 point branch")

    def test_filter_trim_end(self):
        m = Morphology([Branch(np.ones((5, 3)), np.ones(5))])
        split_one = np.ones(5, dtype=bool)
        split_one[4] = 0
        m.label(["test_all"], split_one)
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(4, len(m2), "filtering should return 4 filtered points")
        self.assertEqual(1, len(m2.branches), "expected trim of the end")
        self.assertEqual(4, len(m2.branches[0]), "expected 4 point branch")

    def test_filter_drop_branch(self):
        b = Branch(np.ones((5, 3)), np.ones(5))
        m = Morphology([b])
        m.label(["test_all"])
        b.attach_child(Branch(np.ones((5, 3)), np.ones(5)))
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(5, len(m2), "filtering should return 5 filtered points")
        self.assertEqual(1, len(m2.branches), "expected dropped child")
        self.assertEqual(5, len(m2.branches[0]), "expected 5 point branch")

    def test_filter_skip_dropped(self):
        b = Branch(np.ones((5, 3)), np.ones(5))
        m = Morphology([b])
        m.label(["test_all"])
        c = Branch(np.ones((5, 3)), np.ones(5))
        b.attach_child(c)
        d = Branch(np.ones((5, 3)), np.ones(5))
        d.label(["test_all"])
        c.attach_child(d)
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(10, len(m2), "filtering should return 10 filtered points")
        self.assertEqual(2, len(m2.branches), "expected dropped middle branch")
        self.assertEqual(m2.branches[0], m2.branches[1].parent, "should be connected")

    def test_filter_multiroot(self):
        b = Branch(np.ones((5, 3)), np.ones(5))
        m = Morphology([b])
        c = Branch(np.ones((5, 3)), np.ones(5))
        c.label(["test_all"])
        d = Branch(np.ones((5, 3)), np.ones(5))
        d.label(["test_all"])
        b.attach_child(c)
        b.attach_child(d)
        m2 = m.set_label_filter(["test_all"]).as_filtered()
        self.assertIsNot(m, m2, "filtering should return copy")
        self.assertEqual(10, len(m2), "filtering should return 10 filtered points")
        self.assertEqual(2, len(m2.branches), "expected dropped root branch")
        self.assertTrue(m2.branches[0].is_root, "should be root, root parent is gone")
        self.assertTrue(m2.branches[1].is_root, "should be root, root parent is gone")


class TestRotationSet(unittest.TestCase):
    def setUp(self):
        self.vects = [
            [[0, 1, 0]],
            np.array([[0, 0, 1], [0, 1, 0]]),
            [Rotation.from_rotvec([0, 0, 1])],
            [Rotation.from_rotvec([0, 0, 1]), Rotation.from_rotvec([0, -1, 0])],
        ]
        self.sets = [RotationSet(v) for v in self.vects]

    def test_arrays(self):
        self.assertTrue(np.array(RotationSet(np.empty((2, 3)))).shape, (2, 3))
        self.assertTrue(np.all(np.array(self.sets[0]) == np.array(self.vects[0])))
        self.assertTrue(np.all(np.array(self.sets[1]) == self.vects[1]))
        self.assertTrue(
            np.allclose(np.array(self.sets[2]), np.array([0, 0, 180.0 / np.pi]))
        )
        self.assertTrue(
            np.allclose(
                np.array(self.sets[3]),
                np.array([[0, 0, 180.0 / np.pi], [0, -180.0 / np.pi, 0]]),
            )
        )
        with self.assertRaises(ValueError, msg="It should throw a ValueError") as _:
            RotationSet([])
        with self.assertRaises(ValueError, msg="It should throw a ValueError") as _:
            RotationSet(np.empty((4, 1)))
        with self.assertRaises(ValueError, msg="It should throw a ValueError") as _:
            RotationSet(np.empty((4, 3, 3)))
