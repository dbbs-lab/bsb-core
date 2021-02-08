import unittest, os, sys, numpy as np, h5py
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb._pool import JobPool, FakeFuture
from time import sleep


def test_dud(scaffold, x, y):
    sleep(y)
    return x


class SchedulerBaseTest:
    def test_single_job(self):
        network = Scaffold()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()


@unittest.skipIf(
    MPI.COMM_WORLD.Get_size() < 2, "Skipped during serial testing."
)
class TestParallelScheduler(unittest.TestCase, SchedulerBaseTest):
    def test_double_pool(self):
        network = Scaffold()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        pool.execute()

    def test_master_loop(self):
        network = Scaffold()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        executed = False
        def spy_loop(p):
            nonlocal executed
            executed = True
        pool.execute(master_event_loop=spy_loop)
        if MPI.COMM_WORLD.Get_rank():
            self.assertFalse(executed, 'workers executed master loop')
        else:
            self.assertTrue(executed, 'master loop skipped')

    def test_fake_futures(self):
        network = Scaffold()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 0.1))
        self.assertIs(FakeFuture.done, job._future.done.__func__)
        self.assertFalse(job._future.done())
        self.assertFalse(job._future.running())

    def test_dependencies(self):
        network = Scaffold()
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
            self.assertTrue(result, 'A job with unfinished dependencies was scheduled.')



@unittest.skipIf(
    MPI.COMM_WORLD.Get_size() > 1, "Skipped during parallel testing."
)
class TestSerialScheduler(unittest.TestCase, SchedulerBaseTest):
    pass
