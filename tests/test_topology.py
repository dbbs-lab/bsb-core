from bsb import topology
import unittest, numpy as np


def single_layer():
    r = topology.Region(cls="y_stack", partitions=[])
    c = topology.Partition(type="layer", thickness=150, z_index=0, region=r)
    r._partitions = [c]
    r.arrange(topology.Boundary([0, 0, 0], [100, 100, 100]))
    return r, c


class TestTopology(unittest.TestCase):
    def test_create_topology(self):
        r = topology.Region(partitions=[])
        rb = topology.Region(partitions=[])
        # Workaround for _partitions only being created when references are resolved on
        # root config construction
        r._partitions = []
        rb._partitions = []
        t = topology.create_topology([r])
        self.assertEqual(r, t, "Topology with 1 root region should be the region itself")
        t = topology.create_topology([r, rb])
        self.assertNotEqual(r, t, "Topology with multiple roots should be encapsulated")
        r2 = topology.Region(partitions=[r])
        r2._partitions = [r]
        t = topology.create_topology([r2, r])
        self.assertEqual(r2, t, "Dependency interpreted as additional root")
        r3 = topology.Region(partitions=[r2])
        r3._partitions = [r2]
        t = topology.create_topology([r3, r2, r])
        self.assertEqual(r3, t, "Recursive dependency interpreted as additional root")

    def test_boundary_props(self):
        b = topology.Boundary([0, 0, 0], [100, 100, 100])
        self.assertEqual([0, 0, 0], list(b.ldc))
        self.assertEqual([100, 100, 100], list(b.mdc))
        self.assertEqual(0, b.x)
        self.assertEqual(0, b.y)
        self.assertEqual(0, b.z)
        self.assertEqual(100, b.width)
        self.assertEqual(100, b.height)
        self.assertEqual(100, b.depth)
        # Translate
        b.x = 10
        self.assertEqual(10, b.x)
        self.assertEqual(110, b.mdc[0])
        self.assertEqual(100, b.width)
        # Resize
        b.height = 200
        self.assertEqual(200, b.height)
        self.assertEqual(0, b.y)
        self.assertEqual(200, b.mdc[1])

    def test_cubic(self):
        c = topology.BoxBoundary([0, 0, 0], [100, 100, 100], centered=False)
        ct = topology.BoxBoundary([0, 0, 0], [100, 100, 100], centered=True)
        self.assertEqual([0, 0, 0], list(c.ldc))
        self.assertEqual([100, 100, 100], list(c.mdc))
        # Test centering
        self.assertEqual([-50, -50, -50], list(ct.ldc))
        self.assertEqual([50, 50, 50], list(ct.mdc))

    def test_ystack(self):
        r = topology.Region(cls="y_stack", partitions=[])
        c = topology.Partition(type="layer", thickness=150, z_index=0, region=r)
        c2 = topology.Partition(type="layer", thickness=150, z_index=1, region=r)
        r._partitions = [c2, c]
        r.arrange(topology.Boundary([0, 0, 0], [100, 100, 100]))
        self.assertEqual(0, c.boundaries.y)
        self.assertEqual(150, c2.boundaries.y)
        self.assertEqual(100, c.boundaries.width)
        self.assertEqual(100, c2.boundaries.width)
        self.assertEqual(300, r.boundaries.height)

    def test_partition_chunking(self):
        r, l = single_layer()
        cs = np.array([100, 100, 100])
        # Test 100x150x100 layer producing 2 100x100x100 chunks on top of eachother
        self.assertEqual([[0, 0, 0], [0, 1, 0]], l.to_chunks(cs).tolist())
        # Test translation by whole chunk
        l.boundaries.x += cs[0]
        self.assertEqual([[1, 0, 0], [1, 1, 0]], l.to_chunks(cs).tolist())
        # Translate less than a chunk so that we overflow into an extra layer of x chunks
        l.boundaries.x += 1
        self.assertEqual(
            [[1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], l.to_chunks(cs).tolist()
        )
