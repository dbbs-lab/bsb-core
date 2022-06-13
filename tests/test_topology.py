from bsb import topology
import unittest, numpy as np


def single_layer():
    c = topology.Layer(thickness=150, stack_index=0)
    r = topology.Stack(children=[c])
    topology.create_topology([r], np.array([0, 0, 0]), np.array([100, 100, 100]))
    return r, c


class TestCreateTopology(unittest.TestCase):
    def test_single(self):
        r = topology.Region(name="R", children=[])
        t = topology.create_topology([r], np.array([0, 0, 0]), np.array([100, 100, 100]))
        self.assertEqual(r, t, "Topology with 1 root region should be the region itself")

    def test_unrelated(self):
        r = topology.Region(children=[])
        rb = topology.Region(children=[])
        t = topology.create_topology([r, rb], np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertNotEqual(r, t, "Topology with multiple roots should be encapsulated")

    def test_2gen(self):
        r = topology.Region(children=[])
        r2 = topology.Region(children=[r])
        t = topology.create_topology([r2, r], np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.assertEqual(r2, t, "Dependency interpreted as additional root")

    def test_3gen(self):
        r = topology.Region(children=[])
        r2 = topology.Region(children=[r])
        r3 = topology.Region(children=[r2])
        t = topology.create_topology(
            [r3, r2, r], np.array([0, 0, 0]), np.array([100, 100, 100])
        )
        self.assertEqual(r3, t, "Recursive dependency interpreted as additional root")


class TestTopology(unittest.TestCase):
    def test_stack(self):
        c = topology.Layer(name="c", thickness=150, stack_index=0)
        c2 = topology.Layer(name="c2", thickness=150, stack_index=1)
        r = topology.Stack(name="mystack", children=[c, c2])
        topology.create_topology([r], [0, 0, 0], [100, 100, 100])
        self.assertEqual(0, c.data.y)
        self.assertEqual(150, c2.data.height)
        self.assertEqual(100, c.data.width)
        self.assertEqual(100, c2.data.width)
        self.assertEqual(300, r.data.height)

    def test_partition_chunking(self):
        r, l = single_layer()
        cs = np.array([100, 100, 100])
        # Test 100x150x100 layer producing 2 100x100x100 chunks on top of eachother
        self.assertEqual([[0, 0, 0], [0, 1, 0]], l.to_chunks(cs).tolist())
        # Test translation by whole chunk
        l.data.x += cs[0]
        self.assertEqual([[1, 0, 0], [1, 1, 0]], l.to_chunks(cs).tolist())
        # Translate less than a chunk so that we overflow into an extra layer of x chunks
        l.data.x += 1
        self.assertEqual(
            [[1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], l.to_chunks(cs).tolist()
        )
