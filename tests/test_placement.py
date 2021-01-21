import unittest, os, sys, numpy as np, h5py
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb._pool import JobPool
from time import sleep


def test_dud(scaffold, x, y):
    sleep(y)
    return x


class SchedulerBaseTest:
    def test_single_job(self):
        network = Scaffold()
        pool = JobPool(network)
        job = pool.queue(test_dud, (5, 2))
        pool.execute()


@unittest.skipIf(
    MPI.COMM_WORLD.Get_size() < 2, "Only test parallel scheduler in parallel environment"
)
class TestParallelScheduler(unittest.TestCase, SchedulerBaseTest):
    pass


@unittest.skipIf(
    MPI.COMM_WORLD.Get_size() > 1, "Only test serial scheduler in serial environment"
)
class TestSerialScheduler(unittest.TestCase, SchedulerBaseTest):
    pass
