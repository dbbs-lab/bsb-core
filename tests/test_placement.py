import unittest, os, sys, numpy as np, h5py

from bsb import config
from bsb.connectivity import ConnectionStrategy
from bsb.mixins import NotParallel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.services import MPI
from bsb.core import Scaffold
from bsb.config import Configuration, from_json
from bsb.cell_types import CellType
from bsb.topology import Region, Partition
from bsb.voxels import VoxelSet, VoxelData
from bsb.exceptions import *
from bsb.storage import Chunk
from bsb.placement import PlacementStrategy, RandomPlacement
from bsb.services.pool import JobPool, FakeFuture, create_job_pool
from bsb.unittest import get_config_path, timeout, RandomStorageFixture, NumpyTestCase
from time import sleep


def test_dud(scaffold, x, y):
    sleep(y)
    return x


def test_chunk(scaffold, chunk):
    return chunk


class PlacementDud(PlacementStrategy):
    name = "dud"

    def place(self, chunk, indicators):
        pass


def single_layer_placement(offset=[0.0, 0.0, 0.0]):
    network = Scaffold()
    network.partitions["dud_layer"] = part = Partition(name="dud_layer", thickness=120)
    network.regions["dud_region"] = Region(name="dud_region", children=[part])
    dud_cell = CellType(name="dud", spatial={"count": 40, "radius": 2})
    network.cell_types["dud"] = dud_cell
    dud = PlacementDud(
        name="dud",
        strategy="PlacementDud",
        partitions=[part],
        cell_types=[dud_cell],
        overrides={"dud": {}},
    )
    network.network.origin = offset
    network.placement["dud"] = dud
    network.configuration._bootstrap(network)
    return dud, network


_dud, _net = single_layer_placement()


def _chunk(x, y, z):
    return Chunk((x, y, z), (100, 100, 100))


class TestIndicators(unittest.TestCase):
    def test_cascade(self):
        indicators = _dud.get_indicators()
        dud_ind = indicators["dud"]
        self.assertEqual(2, dud_ind.indication("radius"))
        self.assertEqual(40, dud_ind.indication("count"))
        self.assertEqual(2, dud_ind.get_radius())
        _dud.overrides.dud.radius = 4
        self.assertEqual(4, dud_ind.indication("radius"))
        _dud.overrides.dud.radius = None
        _dud.cell_types[0].spatial.radius = None
        self.assertEqual(None, dud_ind.indication("radius"))
        self.assertRaises(IndicatorError, dud_ind.get_radius)

    def test_guess(self):
        dud, network = single_layer_placement()
        indicators = dud.get_indicators()
        dud_ind = indicators["dud"]
        self.assertEqual(40, dud_ind.guess())
        dud.overrides.dud.count = 400
        self.assertEqual(400, dud_ind.guess())
        bottom_ratio = 1 / 1.2
        bottom = 400 * bottom_ratio / 4
        top_ratio = 0.2 / 1.2
        top = 400 * top_ratio / 4
        for x, y, z in ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(bottom) <= guess <= np.ceil(bottom))
        for x, y, z in ((0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(top) <= guess <= np.ceil(top))
        for x, y, z in ((0, -1, 0), (0, 2, 0), (2, 1, 0), (1, 1, -3)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertEqual(0, guess)

    def test_negative_guess(self):
        dud, network = single_layer_placement(offset=np.array([-300.0, -300.0, -300.0]))
        indicators = dud.get_indicators()
        dud_ind = indicators["dud"]
        bottom_ratio = 1 / 1.2
        bottom = 40 * bottom_ratio / 4
        top_ratio = 0.2 / 1.2
        top = 40 * top_ratio / 4
        for x, y, z in ((-3, -3, -3), (-3, -3, -2), (-2, -3, -3), (-2, -3, -2)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(bottom) <= guess <= np.ceil(bottom))
        for x, y, z in ((-3, -2, -3), (-3, -2, -2), (-2, -2, -3), (-2, -2, -2)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(top) <= guess <= np.ceil(top))
        for x, y, z in ((0, -1, 0), (0, 0, 0), (2, 0, 0), (1, 1, -3)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertEqual(0, guess)


class SchedulerBaseTest:
    @timeout(3)
    def test_create_pool(self):
        pool = create_job_pool(_net)

    @timeout(3)
    def test_single_job(self):
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()

    @timeout(3)
    def test_listeners(self):
        i = 0

        def spy(job):
            nonlocal i
            i += 1

        pool = JobPool(_net, listeners=[spy])
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()
        if not MPI.get_rank():
            self.assertEqual(1, i, "Listeners not executed.")

    def test_placement_job(self):
        pool = JobPool(_net)
        job = pool.queue_placement(_dud, _chunk(0, 0, 0))
        pool.execute()

    def test_chunked_job(self):
        pool = JobPool(_net)
        job = pool.queue_chunk(test_chunk, _chunk(0, 0, 0))
        pool.execute()

    def test_notparallel_ps_job(test):
        spy = 0

        @config.node
        class SerialPStrat(NotParallel, PlacementStrategy):
            def place(self, chunk, indicators):
                nonlocal spy
                test.assertEqual(Chunk([0, 0, 0], None), chunk)
                spy += 1

        pool = JobPool(_net)
        pstrat = _net.placement.add(
            "test", SerialPStrat(strategy="", cell_types=[], partitions=[])
        )
        pstrat.queue(pool, None)
        pool.execute()
        test.assertEqual(1, sum(MPI.allgather(spy)))

    def test_notparallel_cs_job(test):
        spy = 0

        @config.node
        class SerialCStrat(NotParallel, ConnectionStrategy):
            def connect(self, pre, post):
                nonlocal spy

                spy += 1

        pool = JobPool(_net)
        cstrat = _net.connectivity.add(
            "test",
            SerialCStrat(
                strategy="",
                presynaptic={"cell_types": []},
                postsynaptic={"cell_types": []},
            ),
        )
        cstrat.queue(pool)
        pool.execute()
        test.assertEqual(1, sum(MPI.allgather(spy)))


@unittest.skipIf(MPI.get_size() < 2, "Skipped during serial testing.")
class TestParallelScheduler(unittest.TestCase, SchedulerBaseTest):
    @timeout(3)
    def test_double_pool(self):
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()

    @timeout(3)
    def test_master_loop(self):
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        executed = False

        def spy_loop(p):
            nonlocal executed
            executed = True

        pool.execute(master_event_loop=spy_loop)
        if MPI.get_rank():
            self.assertFalse(executed, "workers executed master loop")
        else:
            self.assertTrue(executed, "master loop skipped")

    @timeout(3)
    def test_fake_futures(self):
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        self.assertIs(FakeFuture.done, job._future.done.__func__)
        self.assertFalse(job._future.done())
        self.assertFalse(job._future.running())

    @timeout(3)
    def test_dependencies(self):
        pool = JobPool(_net)
        job = pool.queue(test_dud, (5, 0.1))
        job2 = pool.queue(test_dud, (5, 0.1), deps=[job])
        result = None

        def spy_queue(jobs):
            nonlocal result
            if result is None:
                result = jobs[0]._future.running() and not jobs[1]._future.running()

        pool.execute(master_event_loop=spy_queue)
        if not MPI.get_rank():
            self.assertTrue(result, "A job with unfinished dependencies was scheduled.")

    @unittest.expectedFailure
    def test_notparallel_cs_job(test):
        raise Exception("MPI voodoo deadlocks simple nonlocal assigment")

    @unittest.expectedFailure
    def test_notparallel_ps_job(test):
        raise Exception("MPI voodoo deadlocks simple nonlocal assigment")


@unittest.skipIf(MPI.get_size() > 1, "Skipped during parallel testing.")
class TestSerialScheduler(unittest.TestCase, SchedulerBaseTest):
    pass


class TestPlacementStrategies(
    RandomStorageFixture, NumpyTestCase, unittest.TestCase, engine_name="hdf5"
):
    def test_random_placement(self):
        cfg = from_json(get_config_path("test_single.json"))
        network = Scaffold(cfg, self.storage)
        cfg.placement["test_placement"] = dict(
            strategy="bsb.placement.RandomPlacement",
            cell_types=["test_cell"],
            partitions=["test_layer"],
        )
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertEqual(40, len(ps), "fixed count random placement broken")

    def test_fixed_pos(self):
        cfg = Configuration.default(
            cell_types=dict(test_cell=dict(spatial=dict(radius=2, count=100))),
            placement=dict(
                ch4_c25=dict(
                    strategy="bsb.placement.strategy.FixedPositions",
                    partitions=[],
                    cell_types=["test_cell"],
                )
            ),
        )
        cs = cfg.network.chunk_size
        c4 = [
            Chunk((0, 0, 0), cs),
            Chunk((0, 0, 1), cs),
            Chunk((1, 0, 0), cs),
            Chunk((1, 0, 1), cs),
        ]
        cfg.placement.ch4_c25.positions = pos = MPI.bcast(
            np.vstack(tuple(c * cs + np.random.random((25, 3)) * cs for c in c4))
        )
        network = Scaffold(cfg, self.storage)
        network.compile()
        ps = network.get_placement_set("test_cell")
        pos_sort = pos[np.argsort(pos[:, 0])]
        pspos = ps.load_positions()
        pspos_sort = pspos[np.argsort(pspos[:, 0])]
        self.assertClose(pos_sort, pspos_sort, "expected fixed positions")

    def test_parallel_arrays(self):
        cfg = from_json(get_config_path("test_single.json"))
        network = Scaffold(cfg, self.storage)
        cfg.placement["test_placement"] = dict(
            strategy="bsb.placement.ParallelArrayPlacement",
            cell_types=["test_cell"],
            partitions=["test_layer"],
            spacing_x=50,
            angle=0,
        )
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertEqual(39, len(ps), "fixed count parallel array placement broken")
        pos = ps.load_positions()
        self.assertAll(pos[:, 1] <= cfg.partitions.test_layer.data.mdc[1], "not in layer")
        self.assertAll(pos[:, 1] >= cfg.partitions.test_layer.data.ldc[1], "not in layer")


class TestVoxelDensities(unittest.TestCase):
    def test_particle_vd(self):
        cfg = Configuration.default(
            cell_types=dict(
                test_cell=CellType(spatial=dict(radius=2, density=2, density_key="inhib"))
            ),
            regions=dict(test_region=dict(children=["test_part"])),
            partitions=dict(test_part=dict(type="test")),
            placement=dict(
                voxel_density=dict(
                    strategy="bsb.placement.ParticlePlacement",
                    partitions=["test_part"],
                    cell_types=["test_cell"],
                )
            ),
        )
        network = Scaffold(cfg)
        counts = network.placement.voxel_density.get_indicators()["test_cell"].guess(
            chunk=Chunk([0, 0, 0], [100, 100, 100]),
            voxels=network.partitions.test_part.vs,
        )
        self.assertEqual(4, len(counts), "should have vector of counts per voxel")
        self.assertTrue(np.allclose([78, 16, 8, 27], counts, atol=1), "densities incorr")
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertGreater(len(ps), 90)
        self.assertLess(len(ps), 130)


class VoxelParticleTest(Partition, classmap_entry="test"):
    vs = VoxelSet(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ],
        25,
        data=VoxelData(
            np.array(
                [
                    [0.005, 0.003],
                    [0.001, 0.004],
                    [0.0005, 0.0055],
                    [0.0017, 0.0033],
                ]
            ),
            keys=["inhib", "excit"],
        ),
    )

    def to_chunks(self, chunk_size):
        return [Chunk([0, 0, 0], chunk_size)]

    def chunk_to_voxels(self, chunk):
        return self.vs

    def get_layout(self, hint):
        return hint.copy()

    # Noop city bish, noop noop city bish
    surface = volume = scale = translate = rotate = lambda self, smth: 5
