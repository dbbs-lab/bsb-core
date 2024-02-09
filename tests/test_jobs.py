import unittest
from graphlib import CycleError
from time import sleep

import numpy as np
from bsb_test import (
    NetworkFixture,
    NumpyTestCase,
    RandomStorageFixture,
    skip_parallel,
    timeout,
)

from bsb import config
from bsb.cell_types import CellType
from bsb.config import Configuration
from bsb.connectivity import ConnectionStrategy
from bsb.core import Scaffold
from bsb.mixins import NotParallel
from bsb.placement import FixedPositions, PlacementStrategy, RandomPlacement
from bsb.services import MPI
from bsb.services.pool import Job, JobPool, JobStatus, PoolStatus
from bsb.storage import Chunk
from bsb.topology import Partition


def sleep_y(scaffold, x, y):
    sleep(y)
    return x


def sleep_fail(scaffold, x, y):
    sleep(y)
    return x / 0


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


def create_config():
    @config.node
    class PlacementDud(PlacementStrategy):
        name = "dud"

        def place(self, chunk, indicators):
            pass

    return Configuration.default(
        partitions=dict(dud_layer=Partition(name="dud_layer", thickness=100)),
        cell_types=dict(
            dud_cell=CellType(name="dud", spatial={"count": 1, "radius": 2}),
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


class TestSerialAndParallelScheduler(
    RandomStorageFixture,
    NetworkFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        self.cfg = create_config()
        super().setUp()

    @unittest.skip
    # @timeout(3)
    def test_create_pool(self):
        """todo: test is stuck"""
        pool = self.network.create_job_pool(fail_fast=True)

    #     @timeout(3)
    def test_single_job(self):
        """Test the execution of a single lambda function"""
        pool = JobPool(self.network)
        # Test a single job execution
        job = pool.queue(lambda scaffold, x, y: x * y, (5, 0.1))

        # Check status
        self.assertEqual(job._status, JobStatus.PENDING)

        pool.execute()

        if MPI.get_rank() == 0:
            self.assertEqual(0.5, job.result)
            self.assertEqual(job._status, JobStatus.SUCCESS)

    def test_single_job_fail(self):
        pool = JobPool(self.network)
        # Test if raise errors
        job = pool.queue(lambda scaffold, x, y: x / y, (5, 0))
        pool.execute()
        if not MPI.get_rank():
            self.assertIn("division by zero", str(job._error))
            self.assertEqual(job._status, JobStatus.FAILED)

    def test_multiple_jobs(self):
        """Test the execution of a set of lambda function"""
        pool = JobPool(self.network)
        job1 = pool.queue(lambda scaffold, x, y: x * y, (5, 0.1))
        job2 = pool.queue(lambda scaffold, x, y: x * y, (6, 0.1))
        job3 = pool.queue(lambda scaffold, x, y: x * y, (7, 0.1))
        job4 = pool.queue(lambda scaffold, x, y: x * y, (8, 0.1))

        pool.execute()

        if MPI.get_rank() == 0:
            assert np.isclose(0.5, job1.result)
            assert np.isclose(0.6, job2.result)
            assert np.isclose(0.7, job3.result)
            assert np.isclose(0.8, job4.result)

    @unittest.skip
    def test_cancel_running_job(self):
        """Attempt to cancel a job while running todo: test is stuck"""
        pool = JobPool(self.network)
        job = pool.queue(sleep_y, (5, 0.5))
        pool.execute()
        if MPI.get_rank() == 0:
            with self.assertWarns(Warning) as w:
                job.cancel("Testing")
            self.assertIn("Could not cancel", str(w.warning))

    #     @timeout(3)
    @unittest.skip
    @unittest.skipIf(MPI.get_size() < 2, "Skipped during serial testing.")
    def test_listeners(self):
        """Test that listeners are called and _max_wait is set correctly todo: test is stuck"""
        i = 0
        res = None

        def spy(pool_state, pool_status=None):
            if pool_status != PoolStatus.ENDING:
                nonlocal i
                i += 1

        def collect(pool_state, pool_status=None):
            if pool_state[0]._status == JobStatus.SUCCESS:
                nonlocal res
                res = pool_state[0]._result

        self.network.register_listener(spy, 0.01)
        self.network.register_listener(collect, 0.04)
        pool = self.network.create_job_pool(fail_fast=True)
        job = pool.queue(sleep_y, (5, 0.017))
        pool.execute()
        if not MPI.get_rank():
            self.assertEqual(3, i, "Listeners not executed.")
            self.assertEqual(5, res, "Listeners not executed.")
            self.assertEqual(0.01, pool._max_wait, "_max_wait not properly set.")

    def test_placement_job(self):
        """Test the execution of placement job"""
        self.network.placement.add(
            "test_strat",
            FixedPositions(
                cell_types=["dud_cell"],
                partitions=["dud_layer"],
                positions=[[0, 0, 0]],
            ),
        )

        pool = JobPool(self.network)
        job = pool.queue_placement(
            self.network.placement.test_strat, Chunk((0, 0, 0), (200, 200, 200))
        )
        pool.execute()

        ps = self.network.get_placement_set("dud_cell")
        self.assertClose([[0, 0, 0]], ps.load_positions())


@unittest.skipIf(MPI.get_size() < 2, "Skipped during serial testing.")
class TestParallelScheduler(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.config = create_config()
        super().setUp()

    @unittest.skip
    # @timeout(3)
    def test_double_pool(self):
        """todo: test is stuck"""
        pool = JobPool(self.network)
        pool.queue(sleep_y, (5, 0.1))
        pool.execute()
        pool = JobPool(self.network)
        pool.queue(sleep_y, (5, 0.1))
        pool.execute()

    @unittest.skip
    def test_submitting_closed(self):
        """todo: test is stuck"""
        pool = self.network.create_job_pool(fail_fast=True)
        pool.queue(sleep_y, (5, 0.1))
        pool.execute()

        job = pool.queue(sleep_y, (4, 0.1))
        if MPI.get_rank() == 0:
            with self.assertRaisesRegex(
                RuntimeError, "Attempting to submit job to closed pool."
            ) as err:
                job._enqueue(pool)

    @unittest.skip
    def test_cancel_pending_job(self):
        """Test the cancel method on a job that is not submitted todo: Robin, test is stuck"""
        pool = JobPool(self.network)
        jobs = [pool.queue(sleep_y, (j_id, 0.1)) for j_id in range(6)]
        jobs.append(pool.queue(sleep_y, (100, 0.8), deps=[jobs[5]]))

        pool._job_cancel(jobs[6], "Remove Last One")
        pool.execute()

        self.assertEqual("Remove Last One", str(jobs[6]._error))
        self.assertEqual(JobStatus.CANCELLED, jobs[6]._status)

    @unittest.skip
    def test_cancel_queued_job(self):
        """todo: test is stuck"""
        counter = 0

        def job_killer(job_list, status):
            nonlocal counter
            counter += 1
            if status == PoolStatus.RUNNING and counter == 2:
                job_list[-1].cancel("Testing")

        self.network.register_listener(job_killer, 0.01)
        pool = self.network.create_job_pool(fail_fast=True)
        jobs = [pool.queue(sleep_y, (j_id, 0.1)) for j_id in range(6)]
        jobs.append(pool.queue(sleep_y, (100, 0.8)))
        pool.execute()

        if MPI.get_rank() == 0:
            self.assertEqual(jobs[6]._status, JobStatus.CANCELLED)
            self.assertEqual(str(jobs[6]._error), "Testing")

    @unittest.skip
    # @timeout(3)
    def test_dependencies(self):
        """todo: test is stuck"""
        outcome = None

        def spy(jobs, pool_status):
            nonlocal outcome
            if outcome is None:
                outcome = (
                    "queued" == jobs[0]._status.value
                    and not "queued" == jobs[1]._status.value
                )

        self.network.register_listener(spy, 0.01)
        pool = self.network.create_job_pool(fail_fast=True)
        job = pool.queue(sleep_y, (4, 0.2), submitter={"name": "One"})
        job2 = pool.queue(sleep_y, (5, 0.08), deps=[job], submitter={"name": "Two"})
        job3 = pool.queue(sleep_y, (10, 0.1), submitter={"name": "Three"})

        pool.execute()

        if not MPI.get_rank():
            self.assertTrue(outcome, "A job with unfinished dependencies was scheduled.")
            self.assertEqual(job._result, 4)
            self.assertEqual(job2._result, 5)

    @unittest.skip
    def test_dependency_failure(self):
        """todo: test is stuck"""
        pool = self.network.create_job_pool()
        job = pool.queue(sleep_fail, (4, 0.2), submitter={"name": "One"})
        job2 = pool.queue(sleep_y, (5, 0.1), deps=[job], submitter={"name": "Two"})
        job3 = pool.queue(sleep_y, (4, 0.1), submitter={"name": "Three"})

        # with self.assertRaises(ZeroDivisionError):
        pool.execute()

        if not MPI.get_rank():
            self.assertEqual(str(job2._error), "Job killed for dependency failure")
            self.assertEqual(job3._result, 4)
            self.assertEqual(job._status, JobStatus.FAILED)
            self.assertEqual(job2._status, JobStatus.CANCELLED)
            self.assertEqual(job3._status, JobStatus.SUCCESS)


@skip_parallel
class TestSerialScheduler(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.config = create_config()
        super().setUp()

    def test_notparallel_ps_job(self):
        @config.node
        class SerialPStrat(NotParallel, PlacementStrategy):
            def place(_, chunk, indicators):
                return 1

        pool = JobPool(self.network)
        pstrat = self.network.placement.add(
            "test", SerialPStrat(strategy="", cell_types=[], partitions=[])
        )
        pstrat.queue(pool, None)
        pool.execute()

    def test_notparallel_cs_job(self):
        @config.node
        class SerialCStrat(NotParallel, ConnectionStrategy):
            def connect(_, pre, post):
                return 1

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


class TestSubmissionContext(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.cfg = Configuration.default(
            cell_types=dict(A=dict(spatial=dict(radius=1, count=1))),
            placement=dict(
                test=(
                    PlacementStrategy(
                        strategy="bsb.placement.FixedPositions",
                        cell_types=["A"],
                        partitions=[],
                        positions=[[0, 0, 0]],
                    )
                )
            ),
            connectivity=dict(
                test=(
                    ConnectionStrategy(
                        strategy="bsb.connectivity.AllToAll",
                        presynaptic={"cell_types": ["A"]},
                        postsynaptic={"cell_types": ["A"]},
                    )
                )
            ),
        )
        super().setUp()

    def test_ps_node_submission(self):
        pool = self.network.create_job_pool()
        if pool.is_master():
            self.network.placement.test.queue(pool, [100, 100, 100])
            self.assertEqual(1, len(pool.jobs))
            self.assertEqual("{root}.placement.test", pool.jobs[0].name)

    def test_cs_node_submission(self):
        self.network.run_placement()
        pool = self.network.create_job_pool()
        if pool.is_master():
            self.network.connectivity.test.queue(pool)
            self.assertEqual(1, len(pool.jobs))
            self.assertEqual("{root}.connectivity.test", pool.jobs[0].name)
