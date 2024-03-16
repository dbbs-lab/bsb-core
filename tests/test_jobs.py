import time
import unittest
from graphlib import CycleError
from time import sleep

from bsb_test import (
    NetworkFixture,
    NumpyTestCase,
    RandomStorageFixture,
    skip_parallel,
    timeout,
)

from bsb import (
    MPI,
    CellType,
    Chunk,
    Configuration,
    ConnectionStrategy,
    FixedPositions,
    JobCancelledError,
    JobPoolError,
    NotParallel,
    Partition,
    PlacementStrategy,
    RandomPlacement,
    config,
)
from bsb.services.pool import (
    Job,
    JobStatus,
    PoolProgress,
    PoolProgressReason,
    PoolStatus,
    WorkflowError,
)


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

    @timeout(1)
    def test_create_pool(self):
        with self.network.create_job_pool(quiet=True) as pool:
            pool.execute()

    @timeout(1)
    def test_single_job(self):
        """Test the execution of a single lambda function"""
        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(lambda scaffold, x, y: x * y, (5, 0.1))
            self.assertEqual(job.status, JobStatus.PENDING)

            results = pool.execute(return_results=True)
        if pool.is_main():
            self.assertEqual(0.5, results[job])
            self.assertEqual(job.status, JobStatus.SUCCESS)

    @timeout(1)
    def test_single_job_fail(self):
        """
        Test if a division by zero error is propagated back
        """
        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(lambda scaffold, x, y: x / y, (5, 0))
            with self.assertRaises(WorkflowError):
                pool.execute()
        if pool.is_main():
            self.assertIn("division by zero", str(job._error))
            self.assertEqual(job.status, JobStatus.FAILED)

    @timeout(1)
    def test_multiple_jobs(self):
        """Test the execution of a set of lambda function"""
        with self.network.create_job_pool(quiet=True) as pool:
            job1 = pool.queue(lambda scaffold, x, y: x * y, (5, 0.1))
            job2 = pool.queue(lambda scaffold, x, y: x * y, (6, 0.1))
            job3 = pool.queue(lambda scaffold, x, y: x * y, (7, 0.1))
            job4 = pool.queue(lambda scaffold, x, y: x * y, (8, 0.1))

            results = pool.execute(return_results=True)

        if pool.is_main():
            self.assertAlmostEqual(0.5, results[job1])
            self.assertAlmostEqual(0.6, results[job2])
            self.assertAlmostEqual(0.7, results[job3])
            self.assertAlmostEqual(0.8, results[job4])

    @timeout(1)
    def test_cancel_job(self):
        """
        Cancel a job
        """
        with self.network.create_job_pool(quiet=True) as pool:
            t = time.time()
            job: "Job" = pool.queue(sleep_y, (5, 2))
            job.cancel("Test")
            pool.execute()
        # Confirm the cancellation error
        self.assertEqual(JobCancelledError, type(job.error))
        self.assertEqual("Test", str(job.error))
        # Confirm the job did not run and sleep for 2 seconds
        self.assertLess(t - time.time(), 1)

    @timeout(1)
    def test_cancel_bygone_job(self):
        """
        Attempt to cancel a job after running. Should yield a 'could not cancel' warning.
        """
        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(sleep_y, (5, 0.5))
            pool.execute()
        if pool.is_main():
            with self.assertWarns(Warning) as w:
                job.cancel("Testing")
            self.assertIn("Could not cancel", str(w.warning))

    @timeout(1)
    def test_job_result_before_run(self):
        """Test result exception before the pool has ran"""

        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(sleep_y, (5, 0.5))
        with self.assertRaisesRegex(JobPoolError, "not available"):
            job.result

    @timeout(1)
    def test_job_result_after_run(self):
        """Test result exception after the pool has ran"""

        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(sleep_y, (5, 0.5))
            pool.execute()
        with self.assertRaisesRegex(JobPoolError, "not available"):
            job.result

    @timeout(3)
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

        with self.network.create_job_pool(quiet=True) as pool:
            pool.queue_placement(
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
        self.cfg = create_config()
        super().setUp()

    @timeout(3)
    def test_double_pool(self):
        """Test whether we can open multiple pools sequentially"""
        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(sleep_y, (5, 0.1))
            results = pool.execute(return_results=True)
        if pool.is_main():
            self.assertEqual(5, results[job])
        with self.network.create_job_pool(quiet=True) as pool:
            job = pool.queue(sleep_y, (4, 0.1))
            results = pool.execute(return_results=True)
        if pool.is_main():
            self.assertEqual(4, results[job])

    @timeout(3)
    def test_submitting_closed(self):
        """Test that you can't submit a job after the pool has executed already"""
        with self.network.create_job_pool(quiet=True) as pool:
            pool.execute()
        with self.assertRaises(JobPoolError):
            pool.queue(sleep_y, (4, 0.1))

    @timeout(3)
    def test_cancel_running_job(self):
        """
        Attempt to cancel a job while running. Should yield a 'could not cancel' warning.
        """

        def try_cancel(progress: PoolProgress):
            if (
                progress.reason == PoolProgressReason.POOL_STATUS_CHANGE
                and progress.status == PoolStatus.EXECUTING
            ):
                with self.assertWarnsRegex(Warning, "Could not cancel"):
                    progress.jobs[0].cancel("Test")

        with self.network.create_job_pool(quiet=True) as pool:
            pool.add_listener(try_cancel, 1)
            pool.queue(sleep_y, (5, 0.1))
            pool.execute()

    @timeout(3)
    def test_cancel_queued_job(self):
        """Cancel a job that has been queued, but has not started yet."""

        def job_killer(progress: PoolProgress):
            if (
                progress.reason == PoolProgressReason.POOL_STATUS_CHANGE
                and progress.status == PoolStatus.EXECUTING
            ):
                progress.jobs[-1].cancel("Testing")

        with self.network.create_job_pool(quiet=True) as pool:
            pool.add_listener(job_killer)
            jobs = [pool.queue(sleep_y, (1, 0.001)) for _ in range(200)]
            jobs.append(pool.queue(sleep_y, (1, 0.1)))
            pool.execute()

        if pool.is_main():
            self.assertEqual(jobs[-1].status, JobStatus.CANCELLED)
            self.assertEqual(str(jobs[-1].error), "Testing")

    @timeout(3)
    def test_dependencies(self):
        """
        Test that when the pool starts the jobs without dependencies are queued, and those with dependencies are not.
        """
        outcome = None

        # Add a spy listener that checks that the job with dependencies isn't queued yet
        def spy_initial_pool_queue(progress):
            nonlocal outcome
            if outcome is None:
                outcome = (
                    JobStatus.QUEUED == progress.jobs[0].status
                    and not JobStatus.QUEUED == progress.jobs[1].status
                )

        with self.network.create_job_pool(quiet=True) as pool:
            pool.add_listener(spy_initial_pool_queue)
            job_without_dep = pool.queue(sleep_y, (4, 0.2))
            job_with_dep = pool.queue(sleep_y, (5, 0.08), deps=[job_without_dep])

            results = pool.execute(return_results=True)

        if pool.is_main():
            self.assertTrue(outcome, "A job with unfinished dependencies was scheduled.")
            self.assertEqual(4, results[job_without_dep])
            self.assertEqual(5, results[job_with_dep])

    @timeout(3)
    def test_dependency_failure(self):
        """Test that when a dependency fails, the dependents are cancelled"""
        with self.network.create_job_pool(fail_fast=False, quiet=True) as pool:
            job = pool.queue(sleep_fail, (4, 0.2))
            job2 = pool.queue(sleep_y, (5, 0.1), deps=[job])
            job3 = pool.queue(sleep_y, (4, 0.1))

            try:
                pool.execute()
            except WorkflowError:
                pass

        if not MPI.get_rank():
            self.assertEqual(str(job2.error), "Job killed for dependency failure")
            self.assertEqual(job.status, JobStatus.FAILED)
            self.assertEqual(job2.status, JobStatus.CANCELLED)
            self.assertEqual(job3.status, JobStatus.SUCCESS)

    def test_fail_fast(self):
        """Test that when a single job fails, main raises the error and further execution is aborted."""
        with self.network.create_job_pool(fail_fast=True, quiet=True) as pool:
            job = pool.queue(sleep_fail, (4, 0.01))
            job3 = pool.queue(sleep_y, (4, 0.01))
            job4 = pool.queue(sleep_y, (4, 0.01))
            job5 = pool.queue(sleep_y, (4, 0.01))

            with self.assertRaises(WorkflowError) as workflow_errors:
                pool.execute()
        if pool.is_main():
            self.assertIn(
                ZeroDivisionError,
                [type(e.error) for e in workflow_errors.exception.exceptions],
            )

    @timeout(3)
    def test_listeners(self):
        """Test that listeners are called and max_wait is set correctly"""
        i = 0

        def spy_lt(progress: PoolProgress):
            if progress.reason == PoolProgressReason.MAX_TIMEOUT_PING:
                nonlocal i
                i += 1

        with self.network.create_job_pool(quiet=True) as pool:
            pool.add_listener(spy_lt, 0.1)
            pool.queue(sleep_y, (5, 0.35))
            pool.execute()
        if pool.is_main():
            self.assertEqual(i, 3, "Should have 3 timeout pings")
            self.assertEqual(0.1, pool._max_wait, "_max_wait not properly set.")


@skip_parallel
class TestSerialScheduler(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        self.cfg = create_config()
        super().setUp()

    def test_notparallel_ps_job(self):
        @config.node
        class SerialPStrat(NotParallel, PlacementStrategy):
            def place(_, chunk, indicators):
                return 1

        with self.network.create_job_pool(quiet=True) as pool:
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

        with self.network.create_job_pool(quiet=True) as pool:
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
        with self.network.create_job_pool() as pool:
            if pool.is_main():
                self.network.placement.test.queue(pool, [100, 100, 100])
                self.assertEqual(1, len(pool.jobs))
                self.assertEqual("{root}.placement.test", pool.jobs[0].name)

    @timeout(3)
    def test_cs_node_submission(self):
        self.network.run_placement()
        with self.network.create_job_pool() as pool:
            if pool.is_main():
                self.network.connectivity.test.queue(pool)
                self.assertEqual(1, len(pool.jobs))
                self.assertEqual("{root}.connectivity.test", pool.jobs[0].name)

    @timeout(3)
    def test_no_submitter_submission(self):
        """
        Test that in the absence of submitter information the function name is used.
        """
        with self.network.create_job_pool() as pool:
            if pool.is_main():
                job = pool.queue(sleep_y, (4, 0.2))
                self.assertIn("function sleep_y", job.name)
                self.assertEqual("One", job.context["number"])
