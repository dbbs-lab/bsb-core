from bsb.core import Scaffold
from bsb.unittest import (
    NumpyTestCase,
    FixedPosConfigFixture,
    RandomStorageFixture,
    MorphologiesFixture,
    NetworkFixture,
)
import unittest
import numpy as np
from collections import defaultdict


class TestAllToAll(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.cfg.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"]),
                postsynaptic=dict(cell_types=["test_cell"]),
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(clear=True)

    def test_per_block(self):
        # Test that connections can be stored over chunked layout and can be loaded again.
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        for lchunk, g_itr in cs.nested_iter_connections(direction="out"):
            for gchunk, conns in g_itr:
                ids = conns[0][:, 0]
                self.assertEqual((625,), ids.shape, "625 local_locs per block expected")
                u, c = np.unique(ids, return_counts=True)
                self.assertEqual(25, len(u), "expected exactly 25 local cells")
                self.assertClose(np.arange(0, 25), np.sort(u))
                self.assertClose(25, c)
                ids = conns[1][:, 0]
                self.assertEqual((625,), ids.shape, "625 global_locs per block expected")
                u, c = np.unique(ids, return_counts=True)
                self.assertEqual(25, len(u), "expected exactly 25 global cells")
                self.assertClose(np.arange(0, 25), np.sort(u))
                self.assertClose(25, c)
        self.assertEqual(
            100 * 100, len(self.network.get_connectivity_set("test_cell_to_test_cell"))
        )

    def test_per_local(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        for lchunk in cs.get_local_chunks(direction="out"):
            local_locs, gchunk_ids, global_locs = cs.load_local_connections("out", lchunk)
            ids = local_locs[:, 0]
            self.assertEqual((2500,), ids.shape, "2500 conns per chunk expected")
            u, c = np.unique(ids, return_counts=True)
            self.assertEqual(25, len(u), "expected exactly 25 local cells")
            self.assertClose(np.arange(0, 25), np.sort(u))
            self.assertClose(100, c, "expected 100 global targets per local cell")
            ids = global_locs[:, 0]
            self.assertEqual((2500,), ids.shape, "2500 conns per chunk expected")
            u, c = np.unique(ids, return_counts=True)
            self.assertEqual(25, len(u), "expected exactly 25 global cells")
            self.assertClose(np.arange(0, 25), np.sort(u))
            self.assertClose(100, c, "expected 25 local sources per global cell")
        self.assertEqual(
            100 * 100, len(self.network.get_connectivity_set("test_cell_to_test_cell"))
        )


class TestConnectivitySet(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
    debug=True,
):
    def setUp(self):
        super().setUp()
        self.cfg.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"]),
                postsynaptic=dict(cell_types=["test_cell"]),
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(clear=True)

    def test_load_all(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        data = cs.load_connections()
        try:
            lcol, lloc, gcol, gloc = data
        except (ValueError, TypeError):
            self.fail("`load_connections` did not return 4 args")
        self.assertEqual(10000, len(lcol), "expected full 10k local chunk ids")
        self.assertEqual(10000, len(lloc), "expected full 10k local locs")
        self.assertEqual(10000, len(gcol), "expected full 10k global chunk ids")
        self.assertEqual(10000, len(gloc), "expected full 10k global locs")

    def test_load_local(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        chunks = cs.get_local_chunks("inc")
        data = cs.load_local_connections("inc", chunks[0])
        try:
            lloc, gcol, gloc = data
        except (ValueError, TypeError):
            self.fail("`load_local_connections` did not return 3 args")
        self.assertEqual(2500, len(lloc), "expected full 10k local locs")
        self.assertEqual(2500, len(gcol), "expected full 10k global chunk ids")
        self.assertEqual(2500, len(gloc), "expected full 10k global locs")
        self.assertEqual(4, len(np.unique(gcol)), "Expected data from 4 global chunks")
        self.assertEqual(25, len(np.unique(lloc, axis=0)), "Expected 25 locals")
        unique_globals = len(np.unique(np.hstack((gcol.reshape(-1, 1), gloc)), axis=0))
        self.assertEqual(100, unique_globals, "Expected 100 globals")

    def test_flat_iter(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        itr = cs.flat_iter_connections()
        self.check_a2a_flat_iter(itr, ["inc", "out"], 4, 4)

    def test_nested_iter(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        try:
            iter(cs.nested_iter_connections())
        except TypeError:
            self.fail("expected iteratable")
        dirs = iter(["inc", "out"])
        for dir, local_itr in cs.nested_iter_connections():
            self.assertEqual(next(dirs), dir, "expected `inc` then `out` as direction")
            lchunks = []
            for lchunk, global_itr in local_itr:
                lchunks.append(lchunk)
                gchunks = []
                for gchunk, data in global_itr:
                    gchunks.append(gchunk)
                    try:
                        locals_, globals_ = data
                    except TypeError:
                        self.fail(
                            "`nested_iter_connections` return value should be unpackable"
                        )
                    except ValueError:
                        self.fail("`nested_iter_connections` should return 2 data values")
                    self.assertClose(625, len(locals_), "expected 625 local locs")
                    self.assertClose(625, len(globals_), "expected 625 global locs")
                self.assertEqual(4, len(gchunks), "expected 4 global chunks")
                self.assertEqual(
                    len(gchunks),
                    len(np.unique(gchunks, axis=0)),
                    "each local iter should go to each global chunk exactly once",
                )
            self.assertEqual(4, len(lchunks), "expected 4 local chunks")
            self.assertEqual(
                len(lchunks),
                len(np.unique(lchunks, axis=0)),
                "each dir iter should go to each local chunk exactly once",
            )

    def test_incoming(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        self.check_a2a_flat_iter(iter(cs.incoming), ["inc"], 4, 4)

    def test_outgoing(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        self.check_a2a_flat_iter(iter(cs.outgoing), ["out"], 4, 4)

    def test_from(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(cs.from_(chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(cs.from_(chunks[0])), ["out"], 1, 4)

    def test_to(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(cs.to(chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(cs.to(chunks[0])), ["out"], 4, 1)

    def test_from_to(self):
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(cs.from_(chunks).to(chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(cs.to(chunks).from_(chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(cs.from_(chunks[0]).to(chunks)), ["out"], 1, 4)
        self.check_a2a_flat_iter(iter(cs.to(chunks).from_(chunks[0])), ["out"], 1, 4)
        self.check_a2a_flat_iter(iter(cs.from_(chunks).to(chunks[0])), ["out"], 4, 1)
        self.check_a2a_flat_iter(iter(cs.to(chunks[0]).from_(chunks)), ["out"], 4, 1)
        self.check_a2a_flat_iter(iter(cs.to(chunks[0]).from_(chunks[0])), ["out"], 1, 1)
        self.check_a2a_flat_iter(iter(cs.from_(chunks[0]).to(chunks[0])), ["out"], 1, 1)

    def check_a2a_flat_iter(self, itr, dirs, lcount, gcount):
        self.assertTrue(hasattr(itr, "__next__"), "expected flat iterator")
        spies = defaultdict(lambda: defaultdict(int))
        spies["blocks"] = 0
        spies["block_data"] = []
        while True:
            try:
                data = next(itr)
            except StopIteration:
                break
            except TypeError:
                self.fail("`flat_iter_connections` should be iterable")
            try:
                dir, lchunk, gchunk, block = data
            except TypeError:
                self.fail("`flat_iter_connections` return value should be unpackable")
            except ValueError:
                self.fail("`flat_iter_connections` should return 4 values")
            spies["blocks"] += 1
            spies["dirs"][dir] += 1
            spies["lchunks"][lchunk] += 1
            spies["gchunks"][gchunk] += 1
            spies["block_data"].append(block)
        dircount = len(dirs)
        perdir = lcount * gcount
        blockcount = dircount * perdir
        self.assertEqual(
            blockcount,
            spies["blocks"],
            f"expected {dircount} dir x {lcount} lchunks x {gcount} blocks",
        )
        self.assertEqual(
            sorted(dirs),
            sorted(list(spies["dirs"].keys())),
            f"expected {', '.join(dirs)} blocks",
        )
        for dir in dirs:
            self.assertEqual(
                perdir, spies["dirs"][dir], f"expected {perdir} {dir} blocks"
            )
        local_counts = dict(spies["lchunks"].items())
        self.assertEqual(
            lcount, len(list(local_counts.keys())), f"expected {lcount} local chunks"
        )
        self.assertClose(
            dircount * gcount,
            list(local_counts.values()),
            f"expected each local chunk to occur {dircount} x {gcount} times: {local_counts}",
        )
        global_counts = dict(spies["gchunks"].items())
        self.assertEqual(
            gcount, len(list(global_counts.keys())), f"expected {gcount} global chunks"
        )
        self.assertClose(
            dircount * lcount,
            list(global_counts.values()),
            f"expected each global chunk to occur {dircount} x {lcount} times: {global_counts}",
        )
        self.assertClose(
            2,
            [len(block) for block in spies["block_data"]],
            "expected each block to consist of local and global data",
        )
        self.assertClose(
            625,
            [len(block[0]) for block in spies["block_data"]],
            "expected each block to have 625 local locs",
        )
        self.assertClose(
            625,
            [len(block[1]) for block in spies["block_data"]],
            "expected each block to have 625 global locs",
        )
        return spies


class TestConnWithLabels(
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.cfg.connectivity.add(
            "all_to_all",
            dict(
                strategy="bsb.connectivity.AllToAll",
                presynaptic=dict(cell_types=["test_cell"]),
                postsynaptic=dict(cell_types=["test_cell"]),
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(skip_connectivity=True)
        ps = self.network.get_placement_set("test_cell")
        ps.label(["from_X"], [3, 60, 99])
        self.network.get_placement_set("test_cell").label(["from_Y"], [3, 7, 19])
        self.network.get_placement_set("test_cell").label(["from_F"], [7, 19])
        self.network.get_placement_set("test_cell").label(["Z"], [24])

    def test_from_label(self):
        self.network.connectivity.all_to_all.presynaptic.labels = ["from_X"]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        allcon = cs.load_connections()[0]
        self.assertEqual(300, len(allcon), "should have 3 x 100 cells with from_X label")

    def test_to_label(self):
        self.network.connectivity.all_to_all.postsynaptic.labels = ["from_X"]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        allcon = cs.load_connections()[0]
        self.assertEqual(300, len(allcon), "should have 100 x 3 cells with from_X label")

    def test_dupe_from_labels(self):
        self.network.connectivity.all_to_all.presynaptic.labels = [
            "from_X",
            "from_X",
            "from_Y",
        ]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        allcon = cs.load_connections()[0]
        self.assertEqual(500, len(allcon), "should have 3 x 100 cells with from_X label")

    def test_dupe_labels(self):
        self.network.connectivity.all_to_all.presynaptic.labels = [
            "from_X",
            "from_X",
            "from_Y",
        ]
        self.network.connectivity.all_to_all.postsynaptic.labels = ["from_X", "from_F"]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        allcon = cs.load_connections()[0]
        self.assertEqual(
            (3 + 2) * 5, len(allcon), "should have 3 x 100 cells with from_X label"
        )


class TestConnWithSubCellLabels(
    MorphologiesFixture,
    NetworkFixture,
    FixedPosConfigFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
    morpho_filters=["PurkinjeCell", "StellateCell"],
    debug=True,
):
    def setUp(self):
        super().setUp()
        self.network.connectivity.add(
            "self_intersect",
            dict(
                strategy="bsb.connectivity.VoxelIntersection",
                presynaptic=dict(cell_types=["test_cell"], subcell_labels=["axon"]),
                postsynaptic=dict(cell_types=["test_cell"], subcell_labels=["dendrites"]),
            ),
        )
        print("all morpho", self.network.morphologies.list())
        self.network.cell_types.test_cell.spatial.morphologies = [
            {"names": self.network.morphologies.list()}
        ]
        self.network.compile(skip_connectivity=True)
        print("HELLO?", len(self.network.get_placement_set("test_cell", labels=None)))

    def test_subcell_labels(self):
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("test_cell_to_test_cell")
        allcon = cs.load_connections()[0]
        self.assertEqual(300, len(allcon), "should have 3 x 100 cells with from_X label")
