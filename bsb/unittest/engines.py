from ..exceptions import DatasetNotFoundError, DatasetExistsError
from ..core import Scaffold
from ..cell_types import CellType
from ..config import Configuration
from ..morphologies import Morphology, MorphologySet
from ..storage import Storage, Chunk
from . import (
    NumpyTestCase,
    FixedPosConfigFixture,
    RandomStorageFixture,
    MPI,
    timeout,
    single_process_test,
    get_all_morphology_paths,
    get_morphology_path,
)
import time
import numpy as np
from collections import defaultdict


cfg = Configuration.default(
    cell_types=dict(test_cell=dict(spatial=dict(radius=2, density=1e-3)))
)


class _ScaffoldDummy:
    def __init__(self, cfg):
        self.cfg = self.configuration = cfg

    def get_cell_types(self):
        return list(self.cfg.cell_types.values())


class TestStorage(RandomStorageFixture, engine_name=None):
    @timeout(10)
    def test_init(self):
        # Use the init function to instantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        s = self.storage
        s.create()
        s.init(_ScaffoldDummy(cfg))
        # Test that `init` created the placement sets for each cell type
        for cell_type in cfg.cell_types.values():
            with self.subTest(type=cell_type.name):
                ps = s._PlacementSet(s._engine, cell_type)
                # Test that the placement set is functional after init call
                ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])

    @timeout(10)
    def test_renew(self):
        # Use the renew mechanism to reinstantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        s = self.storage
        s.create()
        self.assertTrue(s.exists())
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        self.assertEqual(
            0,
            len(ps.load_positions()),
            "Data not empty",
        )
        MPI.barrier()
        ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])
        MPI.barrier()
        self.assertEqual(
            MPI.get_size(),
            len(ps.load_positions()),
            "Failure to setup `storage.renew()` test due to chunk reading error.",
        )
        # Spoof a scaffold here, `renew` only requires an object with a
        # `.get_cell_types()` method for its `storage.init` call.
        s.renew(_ScaffoldDummy(cfg))
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        self.assertEqual(
            0,
            len(ps.load_positions()),
            "`storage.renew()` did not clear placement data.",
        )

    @timeout(10)
    def test_move(self):
        s = self.storage
        for _ in range(100):
            os = Storage(self._engine, s.root)
            s.move(s.root[:-5] + "e" + s.root[-5:])
            self.assertTrue(s.exists(), f"{MPI.get_rank()} can't find moved storage yet.")
            self.assertFalse(os.exists(), f"{MPI.get_rank()} still finds old storage.")

    @timeout(10)
    def test_remove_create(self):
        for _ in range(100):
            s = self.storage
            s.remove()
            self.assertFalse(s.exists(), f"{MPI.get_rank()} still finds removed storage.")
            s.create()
            self.assertTrue(s.exists(), f"{MPI.get_rank()} can't find new storage yet.")

    def test_active_config(self):
        s = self.storage
        cfg_a = Configuration.default(regions=dict(a=dict(children=[])))
        cfg_b = Configuration.default(regions=dict(b=dict(children=[])))
        for _ in range(100):
            s.store_active_config(cfg_a)
            expected = s.load_active_config().regions.keys()
            self.assertEqual(cfg_a.regions.keys(), expected, "stored cfg A missmatch")
            s.store_active_config(cfg_b)
            expected = s.load_active_config().regions.keys()
            self.assertEqual(cfg_b.regions.keys(), expected, "stored cfg B missmatch")

    def test_eq(self):
        s = self.storage
        s2 = self.random_storage()
        self.assertEqual(s, s, "Same storage should be equal")
        self.assertNotEqual(s, s2, "Diff storages should be unequal")
        self.assertEqual(s.files, s.files, "Singletons equal")
        self.assertNotEqual(s.files, s2.files, "Diff singletons unequal")
        self.assertNotEqual(s.files, s.morphologies, "Diff singletons unequal")
        self.assertEqual(s.morphologies, s.morphologies, "Singletons equal")
        self.assertNotEqual(s.morphologies, s2.morphologies, "Dff singletons unequal")
        self.assertNotEqual(s.morphologies, "hello", "weird comp should be unequal")


class TestEngine(RandomStorageFixture, engine_name=None):
    def setUp(self):
        super().setUp()
        self.network = Scaffold(storage=self.storage)


class TestPlacementSet(
    FixedPosConfigFixture, RandomStorageFixture, NumpyTestCase, engine_name=None
):
    def setUp(self):
        super().setUp()
        self.network = Scaffold(self.cfg, self.storage)

    def test_init(self):
        ct = self.network.cell_types.test_cell
        ps = self.network.get_placement_set("test_cell")
        self.assertEqual(ct, ps.cell_type, "cell type incorrect")
        self.assertEqual(ct.name, ps.tag, "tag incorrect")
        ct2 = CellType(name="boo", spatial=dict(radius=2, density=1e-3))
        with self.assertRaises(
            DatasetNotFoundError, msg="should raise `DatasetNotFoundError` for unknown PS"
        ):
            self.network.get_placement_set(ct2)

    def test_create(self):
        ct = CellType(name="hehe", spatial=dict(radius=2, density=1e-3))
        if not MPI.get_rank():
            ps = self.storage._PlacementSet.create(self.storage._engine, ct)
            MPI.barrier()
            time.sleep(0.1)
        else:
            MPI.barrier()
            ps = self.storage._PlacementSet(self.storage._engine, ct)
        self.assertEqual("hehe", ps.tag, "tag should be cell type name")
        self.assertEqual(0, len(ps), "new ps should be empty")
        with self.assertRaises(
            DatasetExistsError, msg="creating existing PS should error"
        ):
            ps = self.storage._PlacementSet.create(self.storage._engine, ct)

    def test_exists(self):
        ct = self.network.cell_types.test_cell
        ct2 = CellType(name="hehe", spatial=dict(radius=2, density=1e-3))
        ps = self.network.get_placement_set("test_cell")
        exists = self.storage._PlacementSet.exists
        engine = self.storage._engine
        self.assertTrue(exists(engine, ct), "ps of in cfg ct should exist")
        self.assertFalse(exists(engine, ct2), "ps of random ct should not exist")
        MPI.barrier()
        if not MPI.get_rank():
            ps.remove()
            time.sleep(0.1)
            MPI.barrier()
        else:
            MPI.barrier()
        self.assertFalse(exists(engine, ct), "removed ps should not exist")

    @single_process_test
    def test_require(self):
        ct2 = CellType(name="hehe", spatial=dict(radius=2, density=1e-3))
        # Test that we can create the PS
        self.storage.require_placement_set(ct2)
        # Test that already created PS is not a problem
        self.storage.require_placement_set(ct2)

    def test_clear(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        self.assertEqual(100, len(ps), "expected 100 cells globally")
        ps.clear(chunks=[Chunk([0, 0, 0], self.network.network.chunk_size)])
        MPI.barrier()
        self.assertEqual(75, len(ps), "expected 75 cells after clearing 1 chunk")
        ps.clear()
        MPI.barrier()
        self.assertEqual(0, len(ps), "expected 0 cells after clearing all chunks")

    def test_get_all_chunks(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        self.assertEqual(
            sorted(self.chunks), sorted(ps.get_all_chunks()), "populated chunks incorrect"
        )

    def test_load_positions(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        arr = ps.load_positions()
        self.assertIsInstance(arr, np.ndarray, "Should load pos as numpy arr")
        self.assertEqual((100, 3), arr.shape, "Expected 100x3 position data")
        self.assertEqual(float, arr.dtype, "Expected floats")

    def test_load_no_morphologies(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        with self.assertRaises(DatasetNotFoundError, msg="missing morphos should raise"):
            ps.load_morphologies()

    def test_load_morphologies(self):
        self.network.cell_types.test_cell.spatial.morphologies.append(
            dict(names=["test_cell_A", "test_cell_B"])
        )
        mA = Morphology.from_swc(get_morphology_path("2branch.swc"))
        mB = Morphology.from_swc(get_morphology_path("2comp.swc"))
        self.network.morphologies.save("test_cell_A", mA, overwrite=True)
        self.network.morphologies.save("test_cell_B", mB, overwrite=True)
        for i in range(10):
            ps = self.network.get_placement_set("test_cell")
            with self.assertRaises(DatasetNotFoundError, msg="no morphos yet -> raise"):
                ps.load_morphologies()
            self.network.compile(clear=True)
            ps = self.network.get_placement_set("test_cell")
            ms = ps.load_morphologies()
            self.assertIsInstance(ms, MorphologySet, "Load morpho should return MS")
            self.assertEqual(len(ps), len(ms), "morphos should match amount of pos")
            if "test_cell_A" in ms and "test_cell_B" in ms:
                break
        else:
            self.fail(
                "It seems not all morphologies occur in random morphology distr."
                + "\nShould find: 'test_cell_A', 'test_cell_B'"
                + "\nFound: "
                + ", ".join(
                    f"'{m.meta['name']}'" for m in ms.iter_morphologies(unique=True)
                )
                + "\nAll data:\n["
                + ", ".join(
                    f"'{m.meta['name']}'" for m in ms.iter_morphologies(hard_cache=True)
                )
                + "]"
            )

    def test_load_no_rotations(self):
        self.network.cell_types.test_cell.spatial.morphologies.append(
            dict(names=["test_cell_A", "test_cell_B"])
        )
        mA = Morphology.from_swc(get_morphology_path("2branch.swc"))
        mB = Morphology.from_swc(get_morphology_path("2comp.swc"))
        self.network.morphologies.save("test_cell_A", mA, overwrite=True)
        self.network.morphologies.save("test_cell_B", mB, overwrite=True)
        self.network.compile(clear=True)
        ps = self.network.get_placement_set("test_cell")
        rot = ps.load_rotations()
        self.assertTrue(
            all(np.allclose(np.eye(3), r.as_matrix()) for r in rot),
            "no rotations expected",
        )
        self.assertEqual(len(ps), len(rot), "expected equal amounts of rotations")

    def test_load_rotations(self):
        self.network.cell_types.test_cell.spatial.morphologies.append(
            dict(names=["test_cell_A", "test_cell_B"])
        )
        self.network.placement.ch4_c25.distribute.rotations = dict(strategy="random")
        mA = Morphology.from_swc(get_morphology_path("2branch.swc"))
        mB = Morphology.from_swc(get_morphology_path("2comp.swc"))
        self.network.morphologies.save("test_cell_A", mA, overwrite=True)
        self.network.morphologies.save("test_cell_B", mB, overwrite=True)
        self.network.compile(clear=True)
        ps = self.network.get_placement_set("test_cell")
        rot = ps.load_rotations()
        self.assertTrue(
            any(not np.allclose(np.eye(3), r.as_matrix()) for r in rot),
            "rotations expected",
        )
        self.assertEqual(len(ps), len(rot), "expected equal amounts of rotations")

    def test_4chunks_25cells(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        self.assertEqual(
            100,
            len(ps),
            f"Network was compiled with 100 FixedPositions, but {len(ps)} were placed.",
        )
        for chunk in self.chunks:
            with self.subTest(chunk=chunk):
                with ps.chunk_context([chunk]):
                    self.assertEqual(
                        25,
                        len(ps),
                        "Network was compiled with 25 FixedPositions per chunk,"
                        + f" but {len(ps)} were placed in chunk {chunk}.",
                    )
        pos = self.cfg.placement.ch4_c25.positions
        pos_sort = pos[np.argsort(pos[:, 0])]
        pspos = ps.load_positions()
        pspos_sort = pspos[np.argsort(pspos[:, 0])]
        self.assertClose(
            pos_sort,
            pspos_sort,
            "Network was compiled with FixedPositions,"
            + " but different positions were found.",
        )

    @single_process_test
    def test_chunk_size(self):
        ps = self.network.get_placement_set("test_cell")
        ps.append_data([0, 0, 0], [])
        chunks = ps.get_all_chunks()
        self.assertAll(
            np.isnan([c.dimensions for c in chunks]),
            "Wrote undefined chunk size to PS, so the loaded chunk size should be nan."
            + f" Instead `{chunks[0].dimensions}` was found.",
        )
        ps.append_data(Chunk([0, 0, 1], [20, 20, 20]), [])
        chunks = ps.get_all_chunks()
        self.assertClose(
            20,
            np.array([c.dimensions for c in chunks]),
            "Wrote chunk size `20` to PS with undefined chunk size, so the chunk size"
            + " should be set to the new value."
            + f" Instead `{chunks[0].dimensions}` was found.",
        )

    @single_process_test
    def test_list_input(self):
        ps = self.network.get_placement_set("test_cell")
        try:
            ps.append_data([0, 0, 0], [])
        except Exception:
            self.fail(
                "PlacementSet failed to append `list` typed data. PlacementSets should"
                + " allow this short form to work: `.append_data([0, 0, 0], [])`"
            )
        try:
            ps.append_data([0, 0, 0], [[1, 1, 1]])
        except Exception:
            self.fail(
                "PlacementSet failed to append `list` typed data. PlacementSets should"
                + " allow this short form to work: `.append_data([0, 0, 0], [[1,1,1]])`"
            )
        self.assertEqual(
            1, len(ps), f"PlacementSet placement {len(ps)} after 1 list type input"
        )

    def test_label(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        cells_to_label = [33, 12, 0, 3, 77]
        ps.label(["label1", "label2"], cells_to_label)
        self.assertClose(
            np.sort(cells_to_label),
            ps.get_labelled(["label1"]),
            "Labels should be sort-order match to input",
        )

    def test_label_filter(self):
        if MPI.get_rank():
            MPI.barrier()
        else:
            mA = Morphology.from_swc(get_morphology_path("2comp.swc"))
            self.network.morphologies.save("testA", mA)
            MPI.barrier()
        self.network.cell_types.test_cell.spatial.morphologies = [{"names": ["testA"]}]
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        cells_to_label = [3, 2, 0, 9, 55]
        ps.label(["label1"], cells_to_label)
        cells_to_label = [4, 76, 15, 99, 33]
        ps.label(["label2"], cells_to_label)
        cells_to_label = [4, 76, 15, 9, 55]
        ps.label(["label3"], cells_to_label)
        ps.set_label_filter(["label1"])
        self.assertEqual(5, len(ps), "len should match n labelled cells")
        ps.set_label_filter(["label2"])
        self.assertEqual(5, len(ps), "len should match n labelled cells")
        ps.set_label_filter(["label2", "label2"])
        self.assertEqual(5, len(ps), "double counting")
        ps.set_label_filter(["label1", "label2"])
        self.assertEqual(10, len(ps), "len should match sum n labelled cells")
        ps.set_label_filter(["label3", "label2"])
        self.assertEqual(7, len(ps), "overlapping labels, incorrectly counted")
        self.assertEqual(7, len(ps.load_positions()), "pos not filtered")
        self.assertEqual(7, len(ps.load_morphologies()), "morpho not filtered")
        self.assertEqual(7, len(ps.load_morphologies()), "rot not filtered")


class TestMorphologyRepository(NumpyTestCase, RandomStorageFixture, engine_name=None):
    def setUp(self):
        super().setUp()
        self.mr = self.storage.morphologies

    @single_process_test
    def test_swc_saveload_eq(self):
        for path in get_all_morphology_paths(".swc"):
            with self.subTest(morpho=path.split("/")[-1]):
                m = Morphology.from_swc(path)
                self.mr.save("X", m, overwrite=True)
                lm = self.mr.load("X")
                self.assertEqual(m, lm, "equality violated")
            break

    @single_process_test
    def test_swc_saveload(self):
        for path in get_all_morphology_paths(".swc"):
            with self.subTest(morpho=path.split("/")[-1]):
                m = Morphology.from_swc(path)
                self.mr.save("X", m, overwrite=True)
                lm = self.mr.load("X")
                self.assertEqual(
                    len(m.branches), len(lm.branches), "num branches changed"
                )
                self.assertEqual(
                    m.points.shape,
                    lm.points.shape,
                    f"points shape changed: from {m.points.shape} to {lm.points.shape}",
                )
                self.assertClose(m.points, lm.points, "points changed")
                for i, (b1, b2) in enumerate(zip(m.branches, lm.branches)):
                    self.assertEqual(
                        b1.points.shape,
                        b2.points.shape,
                        f"branch {i} point shape changed",
                    )
                    self.assertClose(b1.points, b2.points, f"branch {i} points changed")

    @single_process_test
    def test_swc_ldc_mdc(self):
        for path in get_all_morphology_paths(".swc"):
            with self.subTest(morpho=path.split("/")[-1]):
                m = Morphology.from_swc(path)
                self.mr.save("pc", m, overwrite=True)
                m = self.mr.load("pc")
                self.assertIn("mdc", m.meta, "missing mdc in loaded morphology")
                self.assertIn("ldc", m.meta, "missing ldc in loaded morphology")


class TestConnectivitySet(
    FixedPosConfigFixture, RandomStorageFixture, NumpyTestCase, engine_name=None
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

    def test_require(self):
        ct = self.network.cell_types.add(
            "new_cell", dict(spatial=dict(radius=2, density=1e-3))
        )
        self.network.require_connectivity_set(
            ct, self.network.cell_types.test_cell, "test"
        )
        self.assertTrue(
            self.storage._ConnectivitySet.exists(self.storage._engine, "test"),
            "must exist after require",
        )

    def test_io(self):
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

    def test_local(self):
        # Test that connections can be stored over chunked layout and can be loaded again.
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

    @single_process_test
    def test_connect_connect(self):
        ct = self.network.cell_types.add(
            "new_cell", dict(spatial=dict(radius=2, density=1e-3))
        )
        self.network.place_cells(ct, [[0, 0, 0], [1, 1, 1], [2, 2, 2]], chunk=[0, 0, 0])
        self.network.place_cells(ct, [[3, 3, 3], [4, 4, 4]], chunk=[0, 0, 1])
        ps0 = self.network.get_placement_set(ct, [[0, 0, 0]])
        ps1 = self.network.get_placement_set(ct, [[0, 0, 1]])
        self.network.get_placement_set(ct)
        cs = self.network.require_connectivity_set(ct, ct, "test")
        cs.connect(ps0, ps1, [], [])
        self.assertEqual(
            0,
            len(cs),
            "After connecting empty data, the ConnectivitySet should remain empty.",
        )
        cs.connect(ps0, ps1, [[1, -1, -1]], [[1, -1, -1]])
        self.assertEqual(
            1, len(cs), "After making 1 connection, the ConnectivitySet should be len 1."
        )
        data = [*cs.flat_iter_connections("out")]
        self.assertEqual(
            1,
            len(data),
            "Wrote 1 connection to set, flat iterator should yield only 1 blockset.",
        )
        self.assertEqual(
            [0, 0, 0],
            data[0][1],
            "Instructed to connect cell 1 of chunk 0 to cell 1 of chunk 1."
            + f" Outgoing chunk should be 0, `{data[0][1]}` found",
        )

    def test_load_all(self):
        cs = self.network.get_connectivity_set("all_to_all")
        data = cs.load_connections()
        try:
            pre, post = data.all()
        except (ValueError, TypeError):
            self.fail("`load_connections` did not return 2 args")
        self.assertEqual(10000, len(pre), "expected full 10k presynaptic locs")
        self.assertEqual(10000, len(post), "expected full 10k postsynaptic locs")

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
