import unittest
from ..exceptions import *
from ..core import Scaffold
from ..cell_types import CellType
from ..config import Configuration
from ..morphologies import Morphology, MorphologySet
from ..storage import Storage, Chunk
from . import (
    NumpyTestCase,
    RandomStorageFixture,
    MPI,
    timeout,
    single_process_test,
    get_all_morphologies,
    get_morphology,
)
import time
import numpy as np


cfg = Configuration.default(
    cell_types=dict(test_cell=dict(spatial=dict(radius=2, density=1e-3)))
)


class _ScaffoldDummy:
    def __init__(self, cfg):
        self.cfg = self.configuration = cfg

    def get_cell_types(self):
        return list(self.cfg.cell_types.values())


class TestStorage(RandomStorageFixture):
    def __init_subclass__(cls, root_factory=None, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory

    def random_storage(self):
        return super().random_storage(root_factory=self._rootf, engine=self._engine)

    @timeout(10)
    def test_init(self):
        # Use the init function to instantiate a storage container to its initial
        # empty state. This test avoids the `Scaffold` object as instantiating it might
        # create or remove data by relying on `renew` or `init` in its constructor.
        s = self.random_storage()
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
        s = self.random_storage()
        s.create()
        self.assertTrue(s.exists())
        ps = s._PlacementSet.require(s._engine, cfg.cell_types.test_cell)
        self.assertEqual(
            0,
            len(ps.load_positions()),
            "Data not empty",
        )
        MPI.Barrier()
        ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])
        MPI.Barrier()
        self.assertEqual(
            MPI.Get_size(),
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
        s = self.random_storage()
        for _ in range(100):
            os = Storage(self._engine, s.root)
            s.move(s.root[:-5] + "e" + s.root[-5:])
            self.assertTrue(s.exists(), f"{MPI.Get_rank()} can't find moved storage yet.")
            self.assertFalse(os.exists(), f"{MPI.Get_rank()} still finds old storage.")

    @timeout(10)
    def test_remove_create(self):
        for _ in range(100):
            s = self.random_storage()
            s.remove()
            self.assertFalse(s.exists(), f"{MPI.Get_rank()} still finds removed storage.")
            s.create()
            self.assertTrue(s.exists(), f"{MPI.Get_rank()} can't find new storage yet.")

    def test_active_config(self):
        s = self.random_storage()
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
        s = self.random_storage()
        s2 = self.random_storage()
        self.assertEqual(s, s, "Same storage should be equal")
        self.assertNotEqual(s, s2, "Diff storages should be unequal")
        self.assertEqual(s.files, s.files, "Singletons equal")
        self.assertNotEqual(s.files, s2.files, "Diff singletons unequal")
        self.assertNotEqual(s.files, s.morphologies, "Diff singletons unequal")
        self.assertEqual(s.morphologies, s.morphologies, "Singletons equal")
        self.assertNotEqual(s.morphologies, s2.morphologies, "Dff singletons unequal")
        self.assertNotEqual(s.morphologies, "hello", "weird comp should be unequal")


class TestEngine(RandomStorageFixture):
    def __init_subclass__(cls, root_factory=None, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory

    def setUp(self):
        self.network = self.random_storage(root_factory=self._rootf, engine=self._engine)


class TestPlacementSet(RandomStorageFixture, NumpyTestCase):
    def __init_subclass__(cls, root_factory=None, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory

    def setUp(self):
        self.storage = self.random_storage(root_factory=self._rootf, engine=self._engine)
        self.cfg = Configuration.default(
            cell_types=dict(test_cell=dict(spatial=dict(radius=2, count=100))),
            placement=dict(
                ch4_c25=dict(
                    strategy="bsb.placement.strategy.FixedPositions",
                    partitions=[],
                    cell_types=["test_cell"],
                )
            ),
        )
        self.chunk_size = cs = self.cfg.network.chunk_size
        self.chunks = [
            Chunk((0, 0, 0), cs),
            Chunk((0, 0, 1), cs),
            Chunk((1, 0, 0), cs),
            Chunk((1, 0, 1), cs),
        ]
        self.cfg.placement.ch4_c25.positions = MPI.bcast(
            np.vstack(
                (
                    np.random.random((25, 3)) * cs + [0, 0, 0],
                    np.random.random((25, 3)) * cs + [0, 0, cs[2]],
                    np.random.random((25, 3)) * cs + [cs[0], 0, 0],
                    np.random.random((25, 3)) * cs + [cs[0], 0, cs[2]],
                )
            )
        )
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
        if not MPI.Get_rank():
            ps = self.storage._PlacementSet.create(self.storage._engine, ct)
            MPI.Barrier()
            time.sleep(0.1)
        else:
            MPI.Barrier()
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
        MPI.Barrier()
        if not MPI.Get_rank():
            ps.remove()
            time.sleep(0.1)
            MPI.Barrier()
        else:
            MPI.Barrier()
        self.assertFalse(exists(engine, ct), "removed ps should not exist")

    @single_process_test
    def test_require(self):
        ct2 = CellType(name="hehe", spatial=dict(radius=2, density=1e-3))
        # Test that we can create the PS
        ps = self.storage.require_placement_set(ct2)
        # Test that already created PS is not a problem
        ps = self.storage.require_placement_set(ct2)

    def test_clear(self):
        self.network.compile()
        ps = self.network.get_placement_set("test_cell")
        self.assertEqual(100, len(ps), "expected 100 cells globally")
        ps.clear(chunks=[Chunk([0, 0, 0], self.network.network.chunk_size)])
        MPI.Barrier()
        self.assertEqual(75, len(ps), "expected 75 cells after clearing 1 chunk")
        ps.clear()
        MPI.Barrier()
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
        ms = ps.load_morphologies()
        # Define final behaviour in #552, for now, just confirm that no data sneaks in.
        self.assertIsInstance(ms, MorphologySet, "missing data should still be MS")
        self.assertEqual(0, len(ms), "there should not be morphology data")

    def test_load_morphologies(self):
        self.network.cell_types.test_cell.spatial.morphologies.append(
            dict(names=["test_cell_A", "test_cell_B"])
        )
        mA = Morphology.from_swc(get_morphology("2branch.swc"))
        mB = Morphology.from_swc(get_morphology("2comp.swc"))
        self.network.morphologies.save("test_cell_A", mA, overwrite=True)
        self.network.morphologies.save("test_cell_B", mB, overwrite=True)
        for i in range(10):
            ps = self.network.get_placement_set("test_cell")
            ms = ps.load_morphologies()
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
        mA = Morphology.from_swc(get_morphology("2branch.swc"))
        mB = Morphology.from_swc(get_morphology("2comp.swc"))
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
        mA = Morphology.from_swc(get_morphology("2branch.swc"))
        mB = Morphology.from_swc(get_morphology("2comp.swc"))
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
        self.assertEqual(100, len(ps), "expected 100 cells globally")
        for chunk in self.chunks:
            with self.subTest(chunk=chunk):
                with ps.chunk_context(chunk):
                    self.assertEqual(25, len(ps), "expected 25 cells per chunk")
        pos = self.cfg.placement.ch4_c25.positions
        pos_sort = pos[np.argsort(pos[:, 0])]
        pspos = ps.load_positions()
        pspos_sort = pspos[np.argsort(pspos[:, 0])]
        self.assertClose(pos_sort, pspos_sort, "expected fixed positions")


class TestMorphologyRepository(NumpyTestCase, RandomStorageFixture):
    def __init_subclass__(cls, root_factory=None, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory

    def setUp(self):
        self.mr = self.random_storage(
            root_factory=self._rootf, engine=self._engine
        ).morphologies

    @single_process_test
    def test_swc_saveload_eq(self):
        for path in get_all_morphologies(".swc"):
            with self.subTest(morpho=path.split("/")[-1]):
                m = Morphology.from_swc(path)
                self.mr.save("X", m, overwrite=True)
                lm = self.mr.load("X")
                self.assertEqual(m, lm, "equality violated")
            break

    @single_process_test
    def test_swc_saveload(self):
        for path in get_all_morphologies(".swc"):
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
                self.assertClose(m.points, lm.points, f"points changed")
                for i, (b1, b2) in enumerate(zip(m.branches, lm.branches)):
                    self.assertEqual(
                        b1.points.shape,
                        b2.points.shape,
                        f"branch {i} point shape changed",
                    )
                    self.assertClose(b1.points, b2.points, f"branch {i} points changed")

    @single_process_test
    def test_swc_ldc_mdc(self):
        for path in get_all_morphologies(".swc"):
            with self.subTest(morpho=path.split("/")[-1]):
                m = Morphology.from_swc(path)
                self.mr.save("pc", m, overwrite=True)
                m = self.mr.load("pc")
                self.assertIn("mdc", m.meta, "missing mdc in loaded morphology")
                self.assertIn("ldc", m.meta, "missing ldc in loaded morphology")
