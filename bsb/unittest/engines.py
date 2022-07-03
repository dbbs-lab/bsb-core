import unittest
from ..config import Configuration
from . import RandomStorageFixture, MPI


cfg = Configuration.default(cell_types=dict(a=dict(spatial=dict(radius=2, density=1e-3))))


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
            ps = storage._PlacementSet.create(storage._engine, cfg.cell_types.a)
            MPI.Barrier()
        else:
            MPI.Barrier()
            ps = storage._PlacementSet(storage._engine, cfg.cell_types.a)
        self.assertEqual("a", ps.tag, "tag should be cell type name")
        self.assertEqual(0, len(ps), "new ps should be empty")
