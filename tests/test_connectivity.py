from bsb.core import Scaffold
from bsb.services import MPI
from bsb.config import Configuration
from bsb.morphologies import Morphology, Branch
from bsb.unittest import (
    NumpyTestCase,
    FixedPosConfigFixture,
    RandomStorageFixture,
    MorphologiesFixture,
    NetworkFixture,
    skip_parallel,
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
        cs = self.network.get_connectivity_set("all_to_all")
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
        self.assertEqual(100 * 100, len(self.network.get_connectivity_set("all_to_all")))

    def test_per_local(self):
        cs = self.network.get_connectivity_set("all_to_all")
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
        self.assertEqual(100 * 100, len(self.network.get_connectivity_set("all_to_all")))


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
        cs = self.network.get_connectivity_set("all_to_all")
        data = cs.load_connections().all()
        try:
            pre, post = data
        except (ValueError, TypeError):
            self.fail("`load_connections` did not return 2 args")
        self.assertEqual(10000, len(pre), "expected full 10k pre locs")
        self.assertEqual(10000, len(post), "expected full 10k post locs")

    def test_load_local(self):
        cs = self.network.get_connectivity_set("all_to_all")
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
        cs = self.network.get_connectivity_set("all_to_all")
        itr = cs.flat_iter_connections()
        self.check_a2a_flat_iter(itr, ["inc", "out"], 4, 4)

    def test_nested_iter(self):
        cs = self.network.get_connectivity_set("all_to_all")
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
        cs = self.network.get_connectivity_set("all_to_all")
        flat = cs.flat_iter_connections
        self.check_a2a_flat_iter(iter(flat(direction="inc")), ["inc"], 4, 4)

    def test_outgoing(self):
        cs = self.network.get_connectivity_set("all_to_all")
        flat = cs.flat_iter_connections
        self.check_a2a_flat_iter(iter(flat(direction="out")), ["out"], 4, 4)

    def test_from(self):
        cs = self.network.get_connectivity_set("all_to_all")
        flat = cs.flat_iter_connections
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(flat("out", chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(flat("out", chunks[0])), ["out"], 1, 4)

    def test_to(self):
        cs = self.network.get_connectivity_set("all_to_all")
        flat = cs.flat_iter_connections
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(flat("out", global_=chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(flat("out", global_=chunks[0])), ["out"], 4, 1)

    def test_from_to(self):
        cs = self.network.get_connectivity_set("all_to_all")
        flat = cs.flat_iter_connections
        chunks = cs.get_local_chunks("inc")
        self.check_a2a_flat_iter(iter(flat("out", chunks, chunks)), ["out"], 4, 4)
        self.check_a2a_flat_iter(iter(flat("out", chunks[0], chunks)), ["out"], 1, 4)
        self.check_a2a_flat_iter(iter(flat("out", chunks, chunks[0])), ["out"], 4, 1)
        self.check_a2a_flat_iter(iter(flat("out", chunks[0], chunks[0])), ["out"], 1, 1)

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
            "expected each local chunk to occur"
            f" {dircount} x {gcount} times: {local_counts}",
        )
        global_counts = dict(spies["gchunks"].items())
        self.assertEqual(
            gcount, len(list(global_counts.keys())), f"expected {gcount} global chunks"
        )
        self.assertClose(
            dircount * lcount,
            list(global_counts.values()),
            "expected each global chunk to occur"
            f" {dircount} x {lcount} times: {global_counts}",
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
        cs = self.network.get_connectivity_set("all_to_all")
        allcon = cs.load_connections().all()[0]
        self.assertEqual(300, len(allcon), "should have 3 x 100 cells with from_X label")

    def test_to_label(self):
        self.network.connectivity.all_to_all.postsynaptic.labels = ["from_X"]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("all_to_all")
        allcon = cs.load_connections().all()[0]
        self.assertEqual(300, len(allcon), "should have 100 x 3 cells with from_X label")

    def test_dupe_from_labels(self):
        self.network.connectivity.all_to_all.presynaptic.labels = [
            "from_X",
            "from_X",
            "from_Y",
        ]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("all_to_all")
        allcon = cs.load_connections().all()[0]
        # 5 cells labelled either X or Y
        self.assertEqual(
            500, len(allcon), "should have 5 x 100 cells with from_X or from_Y label"
        )

    def test_dupe_labels(self):
        self.network.connectivity.all_to_all.presynaptic.labels = [
            "from_X",
            "from_X",
            "from_Y",
        ]
        self.network.connectivity.all_to_all.postsynaptic.labels = ["from_X", "from_F"]
        self.network.compile(append=True, skip_placement=True)
        cs = self.network.get_connectivity_set("all_to_all")
        allcon = cs.load_connections().all()[0]
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
):
    def setUp(self):
        super().setUp()
        self.network.connectivity.add(
            "self_intersect",
            dict(
                strategy="bsb.connectivity.VoxelIntersection",
                presynaptic=dict(cell_types=["test_cell"], morphology_labels=["tag_21"]),
                postsynaptic=dict(
                    cell_types=["test_cell"],
                    morphology_labels=["tag_16", "tag_17", "tag_18"],
                ),
            ),
        )
        self.network.cell_types.test_cell.spatial.morphologies = [
            {"names": self.network.morphologies.list()}
        ]
        self.network.compile(skip_connectivity=True)

    @skip_parallel
    def test_morphology_labels(self):
        f = self.network.connectivity.self_intersect.connect

        def connect_spy(strat, pre, post):
            tc, pre_set = [*pre.placement.items()][0]
            tc, post_set = [*post.placement.items()][0]
            self.assertEqual(
                ["tag_21"],
                pre_set._morphology_labels,
                "expected subcell filters",
            )
            ms = pre_set.load_morphologies()
            m = ms.get(0)
            self.assertEqual(
                len(m),
                len(m.set_label_filter(["tag_21"]).as_filtered()),
                "expected morphology to be filtered already",
            )

            self.assertEqual(
                ["tag_16", "tag_17", "tag_18"],
                post_set._morphology_labels,
                "expected subcell filters",
            )
            ms = post_set.load_morphologies()
            m = ms.get(0)
            self.assertEqual(
                len(m),
                len(m.set_label_filter(["tag_16", "tag_17", "tag_18"]).as_filtered()),
                "expected morphology to be filtered already",
            )
            return f(pre, post)

        conn = self.network.connectivity.self_intersect
        conn.connect = connect_spy.__get__(conn)
        try:
            self.network.compile(append=True, skip_placement=True)
        except Exception as e:
            raise
            self.fail(f"Unexpected error: {e}")
        cs = self.network.get_connectivity_set("self_intersect")
        sloc, dloc = cs.load_connections().all()
        self.assertAll(sloc > -1, "expected only true conn")
        self.assertAll(dloc > -1, "expected only true conn")
        self.assertLess(100, len(cs), "Expected more connections")
        for dir, schunk, gchunk, (sloc, gloc) in cs.flat_iter_connections("out"):
            ps = self.network.get_placement_set("test_cell", chunks=[schunk])
            mset = ps.load_morphologies()
            mids = mset.get_indices(copy=False)[sloc[:, 0]]
            morphos = [*mset.iter_morphologies(unique=True, hard_cache=True)]
            PC = [i for i, m in enumerate(morphos) if m.meta["name"] == "PurkinjeCell"][0]
            self.assertClose(
                [PC],
                np.unique(mids),
                f"expected only PC, stellate found in {schunk} without tag_21",
            )
        for mid, (b, p) in zip(mids, sloc[:, 1:]):
            m = morphos[mid]
            labels = m.branches[b].labels
            self.assertEqual(
                labels.index_of(["tag_21"]),
                labels[p],
                "expected points labelled `tag_21` only",
            )


class TestVoxelIntersection(
    NetworkFixture,
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        self.cfg = Configuration.default(
            cell_types=dict(
                test_cell_A=dict(
                    spatial=dict(radius=1, density=1, morphologies=[dict(names=["A"])])
                ),
                test_cell_B=dict(
                    spatial=dict(radius=1, density=1, morphologies=[dict(names=["B"])])
                ),
            ),
            placement=dict(
                fixed_pos_A=dict(
                    strategy="bsb.placement.FixedPositions",
                    cell_types=["test_cell_A"],
                    partitions=[],
                    positions=[[0, 0, 0], [0, 0, 100], [50, 0, 0], [0, -100, 0]],
                ),
                fixed_pos_B=dict(
                    strategy="bsb.placement.FixedPositions",
                    cell_types=["test_cell_B"],
                    partitions=[],
                    positions=[[95, 0, 0]],
                ),
            ),
        )
        super().setUp()
        self.network.connectivity.add(
            "intersect",
            dict(
                strategy="bsb.connectivity.VoxelIntersection",
                presynaptic=dict(
                    cell_types=["test_cell_A"],
                ),
                postsynaptic=dict(
                    cell_types=["test_cell_B"],
                ),
            ),
        )
        if MPI.get_rank():
            MPI.barrier()
        else:
            mA = Morphology(
                [
                    Branch(
                        [
                            [0, 0, 0],
                            [0, 25, 25],
                            [25, 0, 0],
                            [50, 0, 0],
                        ],
                        [1] * 4,
                    )
                ]
            )
            mA.label(["tip"], [3])
            self.network.morphologies.save("A", mA)
            mB = Morphology(
                [
                    Branch(
                        [
                            [0, 0, 0],
                            [0, 25, 25],
                            [-25, 0, 0],
                            [-50, 0, 0],
                        ],
                        [1] * 4,
                    )
                ]
            )
            mB.label(["top"], [3])
            self.network.morphologies.save("B", mB)
            mC = Morphology(
                [
                    Branch(
                        (
                            b := [
                                [0, 0, 0],
                                [0, 25, 25],
                                [25, 0, 0],
                                [50, 0, 0],
                                [75, 0, 0],
                                [100, 0, 0],
                                [125, 0, 0],
                                [150, 0, 0],
                                [175, 0, 0],
                                [200, 0, 0],
                            ]
                        ),
                        [1] * len(b),
                    )
                ]
            )
            self.network.morphologies.save("C", mC)
            MPI.barrier()

    def test_single_voxel(self):
        # Tests whethervoxel intersection works using a few fixed positions and outcomes.
        self.network.compile()
        cs = self.network.get_connectivity_set("intersect")
        pre_chunks, pre_locs, post_chunks, post_locs = next(
            cs.load_connections().chunk_iter()
        )
        self.assertClose(0, pre_chunks, "expected only conns in base chunk")
        self.assertClose(0, post_chunks, "expected only conns in base chunk")
        self.assertEqual(2, len(pre_locs), "expected 2 connections")
        if not (pre_locs[0].tolist() == [0, 0, 3] and post_locs[0].tolist() == [0, 0, 3]):
            self.fail("expected touching morphologies at their tips in (0,3), (0,3)")
        if not (
            (pre_locs[1].tolist() == [1, 0, 0] and post_locs[1].tolist() == [0, 0, 3])
            or (pre_locs[1].tolist() == [1, 0, 2] and post_locs[1].tolist() == [0, 0, 2])
            or (pre_locs[1].tolist() == [1, 0, 3] and post_locs[1].tolist() == [0, 0, 0])
        ):
            self.fail("expected specific overlap")

    def test_single_voxel_labelled(self):
        # Tests whether a morpho with labels is mapped back to the original points
        self.network.connectivity.intersect.presynaptic.morphology_labels = ["tip"]
        self.network.connectivity.intersect.postsynaptic.morphology_labels = ["top"]
        self.network.compile()
        cs = self.network.get_connectivity_set("intersect")
        pre_chunks, pre_locs, post_chunks, post_locs = next(
            cs.load_connections().chunk_iter()
        )
        self.assertClose(0, pre_chunks, "expected only conns in base chunk")
        self.assertClose(0, post_chunks, "expected only conns in base chunk")
        self.assertEqual(1, len(pre_locs), "expected 1 connection")
        if not (pre_locs[0].tolist() == [0, 0, 3] and post_locs[0].tolist() == [0, 0, 3]):
            self.fail("expected touching morphologies at their tips in (0,3), (0,3)")

    def test_single_voxel_label404(self):
        # Tests whether a morpho without labels is properly excluded
        self.network.cell_types.test_cell_A.spatial.morphologies[0].names.append("C")
        self.network.placement.fixed_pos_A.positions = [[0, 0, 0]] * 2
        self.network.placement.fixed_pos_A.distribute = dict(
            morphologies=dict(strategy="roundrobin")
        )
        self.network.connectivity.intersect.presynaptic.morphology_labels = ["tip"]
        self.network.connectivity.intersect.postsynaptic.morphology_labels = ["top"]
        self.network.compile()
        cs = self.network.get_connectivity_set("intersect")
        pre_chunks, pre_locs, post_chunks, post_locs = next(
            cs.load_connections().chunk_iter()
        )
        self.assertClose(0, pre_chunks, "expected only conns in base chunk")
        self.assertClose(0, post_chunks, "expected only conns in base chunk")
        self.assertEqual(1, len(pre_locs), "expected 1 connection")
        if not (pre_locs[0].tolist() == [0, 0, 3] and post_locs[0].tolist() == [0, 0, 3]):
            self.fail("expected touching morphologies at their tips in (0,3), (0,3)")

    def test_contacts(self):
        mB = Morphology(
            [
                Branch(
                    [
                        [0, 0, 0],
                        [0, 0, 100],
                        [0, 100, 100],
                        [0, 100, 0],
                        [0, 0, 0],
                        [100, 0, 0],
                        [200, 0, 0],
                    ],
                    [1] * 7,
                )
            ]
        )
        self.network.morphologies.save("B", mB, overwrite=True)
        self.network.connectivity.intersect.contacts = 1
        self.network.placement.fixed_pos_A.positions = [[0, 0, 0]]
        self.network.placement.fixed_pos_B.positions = [[0, 0, 0]]
        self.network.cell_types.test_cell_A.spatial.morphologies[0].names = ["C"]
        self.network.compile()
        conns = len(self.network.get_connectivity_set("intersect"))
        self.assertGreater(conns, 0, "no connections formed")
        self.network.connectivity.intersect.contacts = 2
        self.network.compile(clear=True)
        new_conns = len(self.network.get_connectivity_set("intersect"))
        self.assertEqual(conns * 2, new_conns, "Expected double contacts")

    def test_zero_contacts(self):
        self.network.connectivity.intersect.contacts = 0
        self.network.placement.fixed_pos_B.positions = [[100, 0, 0]]
        self.network.cell_types.test_cell_A.spatial.morphologies[0].names = ["C"]
        self.network.compile()
        conns = len(self.network.get_connectivity_set("intersect"))
        self.assertEqual(0, conns, "expected no contacts")
        self.network.connectivity.intersect.contacts = -3
        self.network.compile(clear=True)
        conns = len(self.network.get_connectivity_set("intersect"))
        self.assertEqual(0, conns, "expected no contacts")
