import unittest
from ..config import Configuration
from ..storage import Storage, Chunk
from . import RandomStorageFixture, MPI, timeout
import time


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
        with ps._engine._master_write() as fence:
            fence.guard()
            ps.append_data(Chunk((0, 0, 0), (100, 100, 100)), [0])
        self.assertEqual(
            1,
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


class TestPlacementSet(RandomStorageFixture):
    def __init_subclass__(cls, root_factory=None, *, engine_name, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._engine = engine_name
        cls._rootf = root_factory

    def setUp(self):
        self.network = self.random_storage(root_factory=self._rootf, engine=self._engine)

    def test_create(self):
        storage = self.random_storage()
        if not MPI.Get_rank():
            ps = storage._PlacementSet.create(storage._engine, cfg.cell_types.test_cell)
            MPI.Barrier()
            time.sleep(0.1)
        else:
            MPI.Barrier()
            ps = storage._PlacementSet(storage._engine, cfg.cell_types.test_cell)
        self.assertEqual("test_cell", ps.tag, "tag should be cell type name")
        self.assertEqual(0, len(ps), "new ps should be empty")
