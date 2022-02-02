import unittest
import numpy as np
from bsb.voxels import VoxelSet
from itertools import count as _ico, chain as _ic

class TestVoxelSet(unittest.TestCase):
    def assertClose(self, a, b, **kwargs):
        return self.assertTrue(np.allclose(a, b, **kwargs), f"{a} not {b}")

    def setUp(self):
        self.regulars = [
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 2),
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 2, 2]),
        ]
        self.irregulars = [
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 1, irregular=True),
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 1, 1], irregular=True),
        ]
        self.unequals = [
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1,1,1], [2, 2, 2], [5, 3, 7]]),
            VoxelSet([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[1,1,1], [2, 2, 2], [5, 3, 7]]),
        ]
        self.all = dict(
            _ic(
                zip((f"regular{i}" for i in _ic()), self.regulars),
                zip((f"irregular{i}" for i in _ic()), self.irregulars),
                zip((f"unequal{i}" for i in _ic()), self.unequals),
            )
        )

    def test_voxelset(self):
        # Test empty arrays edge cases
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

    def test_get_size(self):
        for set in self.regulars:
            s = set.get_size(copy=False)
            self.assertTrue(set.get_size(copy=False) is s)
            self.assertFalse(set.get_size(copy=True) is s)
            self.assertEqual(np.ndarray, type(s))
            self.assertClose(2, [1, 1, 1] * s)

        for set in self.irregulars:
            s = set.get_size(copy=False)
            self.assertTrue(set.get_size(copy=False) is s)
            self.assertFalse(set.get_size(copy=True) is s)
            self.assertEqual(np.ndarray, type(s))

    def test_spatial_coords(self):
        set = self.regulars[0]
        self.assertClose([[0, 0, 0], [2, 0, 0], [4, 0, 0]], set.as_spatial_coords())


    def test_get_data(self):
        for label, set in self.all.items():
            self.assertEqual(None, set.get_data(), "No index no data should be None")
            widx = set.get_data([1, 2])
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))
            widx = set.get_data([1, 2], copy=False)
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))
            widx = set.get_data([1, 2], copy=True)
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))

    def test_get_size_matrix(self):
        for label, set in self.all.items():
            self.assertEqual(None, set.get_data(), "No index no data should be None")
            widx = set.get_data([1, 2])
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))
            widx = set.get_data([1, 2], copy=False)
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))
            widx = set.get_data([1, 2], copy=True)
            self.assertEqual(2, len(widx), "index no data should be ndarray of None")
            self.assertEqual(None, widx.dtype.python_type)
            self.assertTrue(all(n is None for n in widx))


    def test_concatenate(self):
        ...
