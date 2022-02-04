import unittest
import numpy as np
import random
from bsb.voxels import VoxelSet
from bsb.storage import Chunk
from bsb.morphologies import Branch, Morphology
from bsb.exceptions import *
from itertools import count as _ico, chain as _ic


class TestVoxelSet(unittest.TestCase):
    def assertClose(self, a, b, msg="", /, **kwargs):
        return self.assertTrue(np.allclose(a, b, **kwargs), f"Expected {a}, got {b}")

    def assertAll(self, a, msg="", /, **kwargs):
        trues = np.sum(a.astype(bool))
        all = np.product(a.shape)
        return self.assertTrue(
            np.all(a, **kwargs), f"{msg}. Only {trues} out of {all} True"
        )

    def setUp(self):
        vs = VoxelSet
        self.regulars = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 2),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 2, 2]),
            vs([[0, 0, 0], [1, 0, 0], [2, 2, 0]], [-1, 2, 2]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], -1),
        ]
        self.irregulars = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 1, irregular=True),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 1, 1], irregular=True),
            vs([[0, 0, 0], [12, 0, 9.4], [2, 0, 0]], [1, -3.5, 1], irregular=True),
            vs([[0, 0, 0], [1, -3.2, 0], [2, 3.6, 0]], [1, -3.5, 1], irregular=True),
        ]
        self.unequals = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 1, 1], [2, 2, 2], [5, 3, 7]]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 1, 1], [2, 2, 2], [5, 3, 7]]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ]
        self.zero_sized = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 0),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, 0, 0]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [2, 2, 2], [5, 3, 7]]),
        ]
        self.dupes = [
            vs([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1),
            vs([[5, 0, 0], [1, 0, 0], [5, 0, 0]], 0),
            vs([[0, -1, 0], [1, 9, 0], [0, -1, 0]], [[0, 2, 0], [2, 2, 2], [5, 3, 7]]),
        ]
        self.empty = vs.empty()
        self.all = dict(
            _ic(
                zip((f"regular_{i}" for i in _ico()), self.regulars),
                zip((f"irregular_{i}" for i in _ico()), self.irregulars),
                zip((f"unequal_{i}" for i in _ico()), self.unequals),
                zip((f"zero_sized_{i}" for i in _ico()), self.zero_sized),
                zip((f"dupes_{i}" for i in _ico()), self.dupes),
                zip((f"empty_{i}" for i in _ico()), (self.empty,)),
            )
        )

        self.data = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 2, [[1], [0], [1]]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 2, 2], [[1], [0], [1]]),
            vs([[0, 0, 0], [1, 0, 0], [2, 2, 0]], [-1, 2, 2], [[1], [0], [1]]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], -1, [[1], [0], [1]]),
            vs([[0, 0, 0], [1, 0, 0], [0, 0, 0]], -1, [[1], [0], [1]]),
        ]

    def test_weird_usage(self):
        arr = np.array([])
        size = np.array([])
        a = VoxelSet(arr, size)
        with self.assertRaises(IndexError):
            a[1]
        with self.assertRaises(IndexError):
            a[1, 1]
        self.assertTrue(a.is_empty)
        with self.assertRaises(AttributeError):
            a.size = 10

    def test_ctor(self):
        with self.assertRaises(ValueError):
            VoxelSet([1, 2, 3], [[1, 0, 0]])
        with self.assertRaises(ValueError):
            VoxelSet([[1, 0, 0, 0]], [[1, 0, 0]])

    def test_data_ctor(self):
        with self.assertRaises(ValueError):
            VoxelSet([], [], [1])
        with self.assertRaises(ValueError):
            VoxelSet([], [], 1)
        VoxelSet([[1, 2, 3]], [[1, 0, 0]], [1])
        VoxelSet([[1, 2, 3]], 1, [1])
        VoxelSet([[1, 2, 3]], 1, [[1, 2]])
        with self.assertWarns(np.VisibleDeprecationWarning):
            VoxelSet([[1, 2, 3], [0, 0, 0]], 1, [[1, 2], [1]])

    def test_unequal_len(self):
        with self.assertRaises(ValueError):
            VoxelSet([[1, 0, 0], [0, 1, 0]], [[1, 0, 0]])
        with self.assertRaises(ValueError):
            VoxelSet([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])

    def test_unbroadcastables(self):
        with self.assertRaises(ValueError):
            unb_size = VoxelSet([[1, 0, 1]], [1, 1, 1, 1])

    def test_ragged(self):
        with self.assertRaises(ValueError):
            with self.assertWarns(np.VisibleDeprecationWarning):
                ragged = VoxelSet([[1, 0, 1], [1, 1]], [1, 1, 1, 1])

    def test_get_size(self):
        for set in self.regulars:
            s = set.get_size(copy=False)
            self.assertTrue(set.get_size(copy=False) is s)
            self.assertTrue(set.get_size(copy=True) is not s)
            self.assertEqual(np.ndarray, type(s))
            self.assertClose(s, [1, 1, 1] * s)
            self.assertClose(set.size, s, "`size` and `get_size` should be equal")

        for set in self.irregulars:
            s = set.get_size(copy=False)
            self.assertTrue(set.get_size(copy=False) is s)
            self.assertFalse(set.get_size(copy=True) is s)
            self.assertEqual(np.ndarray, type(s))
            self.assertClose(set.size, s, "`size` and `get_size` should be equal")

        set = self.empty
        es = set.get_size()
        self.assertEqual((0, 3), es.shape, "Empty size shape incorrect")
        self.assertClose(set.size, es, "`size` and `get_size` should be equal")

    def test_spatial_coords(self):
        set = self.regulars[0]
        self.assertClose([[0, 0, 0], [2, 0, 0], [4, 0, 0]], set.as_spatial_coords())
        set = self.regulars[2]
        self.assertClose([[0, 0, 0], [-1, 0, 0], [-2, 4, 0]], set.as_spatial_coords())
        set = self.irregulars[0]
        self.assertTrue(set.as_spatial_coords() is not set._coords)
        for label, set in self.all.items():
            with self.subTest(label=label):
                sc = set.as_spatial_coords()
                self.assertEqual(2, sc.ndim, "coords should be matrix")
        set = self.empty
        self.assertEqual((0, 3), set.as_spatial_coords().shape, "empty coords not empty")

    def test_as_boxes(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                boxes = set.as_boxes()
                self.assertEqual(np.ndarray, type(boxes), "Boxes not an array.")
                self.assertEqual(2, boxes.ndim, "Boxes not a matrix")
                self.assertEqual(6, boxes.shape[1], "Boxes not 6 cols")
                self.assertAll(boxes[:, :3] <= boxes[:, 3:], "Boxes not minmax")

    def test_as_boxtree(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                set.as_boxtree()

    def test_get_data(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                self.assertEqual(None, set.get_data(), "No index no data should be None")
                self.assertEqual(None, set.get_data([1, 2]), "No data should be None")
                self.assertEqual(None, set.data, "No data should be None")

    def test_snap_to_grid(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                vs = set.snap_to_grid([1, 1, 1])
                self.assertEqual(len(set), len(vs), "voxels dropped")
                vs = set.snap_to_grid([1, 1, 0])
                self.assertTrue(vs.regular, "snap to grid should make regular grid")
                self.assertClose(0, vs.raw[:, 2], "0 width dimension should be flat")
        for set in self.data:
            vs = set.snap_to_grid([1, 1, 1])
            self.assertEqual(set.data.shape, vs.data.shape, "snap to grid changed data")
            self.assertClose(set.data, vs.data, "snap to grid changed data")

    def test_snap_unique(self):
        for set in self.dupes:
            vs = set.snap_to_grid([1, 1, 1])
            self.assertEqual(len(set), len(vs), "voxels dropped without unique")
            vs = set.snap_to_grid([1, 1, 1], unique=True)
            self.assertNotEqual(len(set), len(vs), "no duplicates dropped with unique")
        for set in self.data:
            vs = set.snap_to_grid([1, 1, 1], unique=True)
            self.assertEqual(len(vs.raw), len(vs.data), "data wrong with unique")

    def test_get_size_matrix(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                sm = set.get_size_matrix(copy=False)
                self.assertClose(set.get_raw(copy=False).shape, sm.shape)

    def test_of_equal_size(self):
        self.assertFalse(self.unequals[2]._single_size, "not internally equal")
        self.assertTrue(self.unequals[2].of_equal_size, "but actually equal")

    def test_raw(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                # Test that it functions for all types
                r = set.get_raw(copy=False)
                self.assertIsNot(r, set.raw, "copy failed")

    def test_iter(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                y = set.get_raw(copy=False)
                for i, vox in enumerate(set):
                    self.assertClose(vox, y[i], "voxel iteration")

    def test_bool(self):
        self.assertTrue(self.empty.is_empty, "Empty not empty")
        self.assertFalse(self.empty, "Empty set should not be True")
        for label, set in self.all.items():
            if label.startswith("empty"):
                continue
            with self.subTest(label=label):
                self.assertFalse(set.is_empty, "non empty set empty")
                self.assertTrue(set, "non empty set False")

    def test_from_flat_morphology(self):
        branches = [
            Branch([i, i, i, i, i], [0, 1, 2, 3, 4], [0] * 5, [1] * 5) for i in range(5)
        ]
        morpho = Morphology(branches)
        vs = morpho.voxelize(16)
        self.assertLess(0, len(vs), "Empty voxelset from non empty morpho")
        self.assertClose(0, vs.get_raw(copy=False)[:, 2], "Flat morphology not flat VS")
        data = vs.get_data()
        vs = VoxelSet.from_morphology(morpho, 16, with_data=False)
        self.assertLess(0, len(vs), "Empty voxelset from non empty morpho")
        self.assertClose(0, vs.get_raw(copy=False)[:, 2], "Flat morphology not flat VS")

    def test_from_empty_morphology(self):
        empty_morpho = Morphology([])
        vs = empty_morpho.voxelize(16)
        self.assertEqual(0, len(vs), "Non-empty voxelset from empty morpho")

    def test_from_point_morphology(self):
        point_morpho = Morphology([Branch([100], [1], [100], [1])])
        vs = point_morpho.voxelize(16)
        self.assertEqual(1, len(vs), "Point morpho")
        self.assertClose(0, vs.get_raw(copy=False))

    def test_select(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                vs = set.select([0, 0, 0], [0, 0, 0])
                self.assertEqual(0, len(vs), "empty select nonempty set")
                vs = set.select([-1000, -1000, -1000], [1000, 1000, 1000])
                self.assertEqual(len(set), len(vs), "big select didnt select all")
        # vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 2, 2]),
        vs = self.regulars[1]
        res = vs.select([-1, -1, -1], [3.5, 0.5, 0.5])
        self.assertEqual(2, len(res), "unexpected selection")
        self.assertClose(0, res.raw[0], "unexpected selection")
        self.assertClose([1, 0, 0], res.raw[1], "unexpected selection")
        c = Chunk([0, 0, 0], [100, 100, 100])
        res = vs.select_chunk(c)
        self.assertEqual(len(res), len(vs), "big chunk didnt select all")

    def test_resize(self):
        vs = self.regulars[0]
        _vs = vs.copy()
        vs.resize([1, 1, 1])
        self.assertClose(vs.as_spatial_coords() * 2, _vs.as_spatial_coords(), "not half")
        self.assertTrue(vs.regular, "still regular")
        with self.assertRaises(ValueError):
            vs.resize(None)
        with self.assertRaises(ValueError):
            vs.resize([[3, 3, 3], [3, 3, 3]])
        vs.resize([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        self.assertFalse(vs.regular, "of equal size, but not regular anymore")

    def test_empty(self):
        self.assertEqual(0, len(self.empty))
        self.assertFalse(self.empty)
        self.assertEqual((0, 3), self.empty.get_size_matrix().shape)

    def test_empty_concat(self):
        vs = VoxelSet.concatenate(self.empty)

    def test_double_empty_concat(self):
        vs = VoxelSet.concatenate(self.empty, self.empty)

    def test_data_concat(self):
        vs = VoxelSet.concatenate(self.data[3], self.regulars[2])

    def test_bounds(self):
        for label, set in self.all.items():
            if not label.startswith("empty"):
                with self.subTest(label=label):
                    bounds = set.bounds
                    self.assertIs(bounds, set.bounds, "bounds should be cached")
                    self.assertEqual(2, len(bounds), "should be tuple of min max")
                    self.assertIs(tuple, type(bounds), "should be tuple")
                    self.assertEqual(3, len(bounds[0]), "3 dim")
                    self.assertEqual(3, len(bounds[1]), "3 dim")
                    self.assertTrue(all(a <= b for a, b in zip(*bounds)), "min max")
        with self.assertRaises(EmptyVoxelSetError):
            self.empty.bounds
        vs = self.unequals[0]

    def test_one(self):
        vs = VoxelSet.one([100, 0, 0], [120, 20, 20])
        self.assertEqual(1, len(vs), "voxelset with single voxel should be len 1")
        b = vs.bounds
        self.assertClose([100, 0, 0], b[0], "incorr min bounds")
        self.assertClose([120, 20, 20], b[1], "incorr max bounds")
        vs = VoxelSet.one([120, 20, 20], [100, 0, 0])
        b = vs.bounds
        self.assertClose([100, 0, 0], b[0], "incorr min bounds")
        self.assertClose([120, 20, 20], b[1], "incorr max bounds")
        vs = VoxelSet.one([[100, 0, 0]], [120, 20, 20])
        with self.assertRaises(ValueError):
            vs = VoxelSet.one([100, 0, 0, 0], [120, 20, 20])
        vs = VoxelSet.one([[100, 0, 0]], [120, 20, 20], 1)
        self.assertEqual(1, vs.get_data(0))
        vs = VoxelSet.one([[100, 0, 0]], [120, 20, 20], [1])
        self.assertEqual(1, vs.get_data(0))
        vs = VoxelSet.one([[100, 0, 0]], [120, 20, 20], [1, 1])
        self.assertEqual([1, 1], list(vs.get_data(0)))

    def test_index(self):
        pass
        # for label, set in self.all.items():
        #     vs = set[0:0, 0, 0, 0]
        #     print(set[0:0, 0])
        #     print(len(vs), vs.get_raw(copy=False).shape)
        #     with self.assertRaises(IndexError):
        #         vs = set[0:0, 0]
        #     if not vs.is_empty:
        #         print(vs)
        #         print(len(vs), vs.get_raw(copy=False).shape)

    def test_copy(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                self.assertIsNot(set, set.copy(), "copy failed")
                self.assertIsNot(
                    set.get_raw(copy=False), set.copy().get_raw(copy=False), "copy failed"
                )

    def test_concatenate(self):
        for i in range(1000):
            choices = random.choices(list(self.all.items()), k=random.randint(0, 5))
            n = len(choices)
            labels = [lbl for lbl, set_ in choices]
            sets = [set_ for lbl, set_ in choices]
            with self.subTest(labels=labels):
                vs = VoxelSet.concatenate(*sets)
