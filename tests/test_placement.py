import unittest, os, sys, numpy as np, h5py
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb.objects import CellType
from bsb.placement import PlacementStrategy
from bsb._pool import JobPool, FakeFuture, create_job_pool
from test_setup import timeout
from time import sleep


def test_dud(scaffold, x, y):
    sleep(y)
    return x


def test_chunk(scaffold, chunk, chunk_size):
    return chunk


class PlacementDud(PlacementStrategy):
    name = "dud"

    def place(self, chunk, chunk_size, indicators):
        pass


network = Scaffold()
dud_cell = CellType(name="dud", spatial={"count": 40, "radius": 2})
network.cell_types["dud"] = dud_cell
dud = PlacementDud(name="dud", cls="PlacementDud", partitions=[], cell_types=[])
network.placement["dud"] = dud
dud._cell_types = [dud_cell]


class SchedulerBaseTest:
    @timeout(3)
    def test_create_pool(self):
        pool = create_job_pool(network)

    @timeout(3)
    def test_single_job(self):
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()

    @timeout(3)
    def test_listeners(self):
        i = 0

        def spy(job):
            nonlocal i
            i += 1

        pool = JobPool(network, listeners=[spy])
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()
        if not MPI.COMM_WORLD.Get_rank():
            self.assertEqual(1, i, "Listeners not executed.")

    def test_placement_job(self):
        pool = JobPool(network)
        job = pool.queue_placement(dud, [0, 0, 0], (100, 100, 100))
        pool.execute()

    def test_chunked_job(self):
        pool = JobPool(network)
        job = pool.queue_chunk(test_chunk, [0, 0, 0], (100, 100, 100))
        pool.execute()


@unittest.skipIf(MPI.COMM_WORLD.Get_size() < 2, "Skipped during serial testing.")
class TestParallelScheduler(unittest.TestCase, SchedulerBaseTest):
    @timeout(3)
    def test_double_pool(self):
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()

    @timeout(3)
    def test_master_loop(self):
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        executed = False

        def spy_loop(p):
            nonlocal executed
            executed = True

        pool.execute(master_event_loop=spy_loop)
        if MPI.COMM_WORLD.Get_rank():
            self.assertFalse(executed, "workers executed master loop")
        else:
            self.assertTrue(executed, "master loop skipped")

    @timeout(3)
    def test_fake_futures(self):
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        self.assertIs(FakeFuture.done, job._future.done.__func__)
        self.assertFalse(job._future.done())
        self.assertFalse(job._future.running())

    @timeout(3)
    def test_dependencies(self):
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        job2 = pool.queue(test_dud, (5, 0.1), deps=[job])
        result = None

        def spy_queue(jobs):
            nonlocal result
            if result is None:
                result = jobs[0]._future.running() and not jobs[1]._future.running()

        pool.execute(master_event_loop=spy_queue)
        if not MPI.COMM_WORLD.Get_rank():
            self.assertTrue(result, "A job with unfinished dependencies was scheduled.")


@unittest.skipIf(MPI.COMM_WORLD.Get_size() > 1, "Skipped during parallel testing.")
class TestSerialScheduler(unittest.TestCase, SchedulerBaseTest):
    pass


class TestPlacementStrategies(unittest.TestCase):
    pass
