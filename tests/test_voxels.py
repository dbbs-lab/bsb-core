import unittest
import numpy as np
from bsb.voxels import VoxelSet
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
        self.empty = vs.empty()
        self.all = dict(
            _ic(
                zip((f"regular_{i}" for i in _ico()), self.regulars),
                zip((f"irregular_{i}" for i in _ico()), self.irregulars),
                zip((f"unequal_{i}" for i in _ico()), self.unequals),
                zip((f"zero_sized_{i}" for i in _ico()), self.zero_sized),
                zip((f"empty_{i}" for i in _ico()), (self.empty,)),
            )
        )

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

    def test_get_size_matrix(self):
        for label, set in self.all.items():
            with self.subTest(label=label):
                sm = set.get_size_matrix(copy=False)
                self.assertClose(set.raw(copy=False).shape, sm.shape)

    def test_of_equal_size(self):
        self.assertFalse(self.unequals[2]._single_size, "not internally equal")
        self.assertTrue(self.unequals[2].of_equal_size, "but actually equal")

    def test_concatenate(self):
        ...
