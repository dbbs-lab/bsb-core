import inspect
import unittest

import bsb_test

from bsb import VoxelSet


class TestVoxelSet(bsb_test.NumpyTestCase, unittest.TestCase):
    def setUp(self):
        vs = VoxelSet
        self.regulars = [
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 2),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [2, 2, 2]),
            vs([[0, 0, 0], [1, 0, 0], [2, 2, 0]], [-1, 2, 2]),
            vs([[0, 0, 0], [1, 0, 0], [2, 0, 0]], -1),
        ]

    def test_boxtree(self):
        tree = self.regulars[0].as_boxtree()
        gen = tree.query([(0, 0, 0, 3, 0, 0), (300, 0, 0, 300, 0, 0)])
        self.assertTrue(inspect.isgenerator(gen), "boxtree queries should return gen")
        res = list(gen)
        self.assertEqual([[0, 1], []], res, "incorrect results")

    def test_boxtree_unique(self):
        tree = self.regulars[0].as_boxtree()
        gen = tree.query(
            [(0, 0, 0, 3, 0, 0), (300, 0, 0, 300, 0, 0), (3, 0, 0, 6, 0, 0)], unique=True
        )
        self.assertTrue(
            inspect.isgenerator(gen), "unique boxtree queries should return gen"
        )
        res = list(gen)
        self.assertEqual([0, 1, 2], res, "incorrect results")
