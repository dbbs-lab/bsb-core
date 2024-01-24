import unittest
from graphlib import CycleError
from time import sleep

from bsb_test import timeout

from bsb import config
from bsb.cell_types import CellType
from bsb.config import Configuration
from bsb.connectivity import ConnectionStrategy
from bsb.core import Scaffold
from bsb.mixins import NotParallel
from bsb.placement import PlacementStrategy, RandomPlacement
from bsb.services import MPI
from bsb.services.pool import JobPool, JobsListener
from bsb.storage import Chunk
from bsb.topology import Partition


def sleep_y(scaffold, x, y):
    sleep(y)
    return x


class TestDependencyOrder(unittest.TestCase):
    def test_sort_order(self):
        a = RandomPlacement(cell_types=[], partitions=[], name="A")
        b = RandomPlacement(cell_types=[], partitions=[], name="B")
        c = RandomPlacement(cell_types=[], partitions=[], name="C")
        self.assertEqual([a, b, c], sorted([c, b, a]), "should sort by name")

    def test_dependency_order(self):
        b = RandomPlacement(cell_types=[], partitions=[], name="B")
        c = RandomPlacement(cell_types=[], partitions=[], name="C")
        a = RandomPlacement(cell_types=[], partitions=[], name="A", depends_on=[c])
        self.assertEqual(
            [b, c, a], a.sort_deps([a, b, c]), "should sort by deps, then by name"
        )

    def test_missing_dependency(self):
        b = RandomPlacement(cell_types=[], partitions=[], name="B")
        c = RandomPlacement(cell_types=[], partitions=[], name="C")
        a = RandomPlacement(cell_types=[], partitions=[], name="A", depends_on=[c])
        self.assertEqual([a, b], a.sort_deps([a, b]), "should not insert missing deps")

    def test_cyclical_error(self):
        b = RandomPlacement(cell_types=[], partitions=[], name="B")
        c = RandomPlacement(cell_types=[], partitions=[], name="C")
        a = RandomPlacement(cell_types=[], partitions=[], name="A", depends_on=[c])
        c.depends_on = [a]
        with self.assertRaises(CycleError):
            a.sort_deps([a, b, c])


def create_network():
    @config.node
    class PlacementDud(PlacementStrategy):
        name = "dud"

        def place(self, chunk, indicators):
            pass

    return Scaffold(
        Configuration.default(
            partitions=dict(dud_layer=Partition(name="dud_layer", thickness=120)),
            cell_types=dict(
                dud_cell=CellType(name="dud", spatial={"count": 40, "radius": 2}),
            ),
            placement=dict(
                dud=PlacementDud(
                    name="dud",
                    strategy=PlacementDud,
                    partitions=["dud_layer"],
                    cell_types=["dud_cell"],
                    overrides={"dud": {}},
                )
            ),
        )
    )


class TestSerialAndParallelScheduler(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.network = create_network()

    # @timeout(3)
    def test_create_pool(self):
        pool = self.network.create_job_pool()

    #     @timeout(3)
    def test_single_job(self):
        pool = JobPool(self.network)
        job = pool.queue_function(lambda scaffold, x, y: x * y, (5, 0.1))
        pool.execute()
        self.assertEqual(0.5, job.result)

    #     @timeout(3)
    def test_listeners(self):
        i = 0

        def spy(pool_state):
            nonlocal i
            i += 1

        self.network.register_listener(spy, max_wait=0.01)
        pool = self.network.create_job_pool()
        job = pool.queue_function(sleep_y, (5, 0.019))
        pool.execute()
        if not MPI.get_rank():
            self.assertEqual(2, i, "Listeners not executed.")

    def test_placement_job(self):
        pool = JobPool(self.network)
        job = pool.queue_placement(
            self.network.placement.dud, Chunk((0, 0, 0), (100, 100, 100))
        )
        pool.execute()

    def test_notparallel_ps_job(self):
        spy = 0

        @config.node
        class SerialPStrat(NotParallel, PlacementStrategy):
            def place(_, chunk, indicators):
                nonlocal spy
                self.assertEqual(Chunk([0, 0, 0], None), chunk)
                spy += 1

        pool = JobPool(self.network)
        pstrat = self.network.placement.add(
            "test", SerialPStrat(strategy="", cell_types=[], partitions=[])
        )
        pstrat.queue(pool, None)
        pool.execute()
        self.assertEqual(1, sum(MPI.allgather(spy)))

    def test_notparallel_cs_job(self):
        spy = 0

        @config.node
        class SerialCStrat(NotParallel, ConnectionStrategy):
            def connect(_, pre, post):
                nonlocal spy

                spy += 1

        pool = JobPool(self.network)
        cstrat = self.network.connectivity.add(
            "test",
            SerialCStrat(
                strategy="",
                presynaptic={"cell_types": []},
                postsynaptic={"cell_types": []},
            ),
        )
        cstrat.queue(pool)
        pool.execute()
        self.assertEqual(1, sum(MPI.allgather(spy)))


@unittest.skipIf(MPI.get_size() < 2, "Skipped during serial testing.")
class TestParallelScheduler(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.network = create_network()

    # @timeout(3)
    def test_double_pool(self):
        pool = JobPool(self.network)
        job = pool.queue(sleep_y, (5, 0.1))
        pool.execute()
        pool = JobPool(self.network)
        job = pool.queue(sleep_y, (5, 0.1))
        pool.execute()

    # @timeout(3)
    def test_dependencies(self):
        class Spy(JobsListener):
            def __init__(self, rt):
                super().__init__(refresh_time=rt)

            def receive(self, jobs):
                nonlocal result
                if result is None:
                    result = "Running" in jobs[0] and not "Running" in jobs[1]

        pool = JobPool(self.network, listeners=[Spy(0.001)])
        job = pool.queue_function(sleep_y, (5, 0.1))
        job2 = pool.queue_function(sleep_y, (5, 0.1), deps=[job])
        result = None

        pool.execute()
        if not MPI.get_rank():
            self.assertTrue(result, "A job with unfinished dependencies was scheduled.")

    @unittest.expectedFailure
    def test_notparallel_cs_job(test):
        raise Exception("MPI voodoo deadlocks simple nonlocal assigment")

    @unittest.expectedFailure
    def test_notparallel_ps_job(test):
        raise Exception("MPI voodoo deadlocks simple nonlocal assigment")


@unittest.skipIf(MPI.get_size() > 1, "Skipped during parallel testing.")
class TestSerialScheduler(unittest.TestCase):
    pass
