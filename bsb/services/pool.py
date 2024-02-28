"""
Job pooling module.

Jobs derive from the base :class:`.Job` class which can be put on the queue of a
:class:`.JobPool`. In order to submit themselves to the pool Jobs will
:meth:`~.Job.serialize` themselves into a predefined set of variables::

    job.serialize() -> (job_type, f, args, kwargs)

* ``job_type`` should be a string that is a class name defined in this module.
  (e.g. ``"PlacementJob")
* ``f`` should be the function object that the job's ``execute`` method should
  execute.
* ``args`` and ``kwargs`` are the args to be passed to that ``f``.

The :meth:`.Job.execute` handler can help interpret ``args`` and ``kwargs``
before running ``f``. The execute handler has access to the scaffold on the MPI
process so one best serializes just the name of some part of the configuration,
rather than trying to pickle the complex objects. For example, the
:class:`.PlacementJob` uses the first ``args`` element to store the
:class:`~bsb.placement.PlacementStrategy` name and then retrieve it from the
scaffold:

.. code-block:: python

    @staticmethod
    def execute(job_owner, f, args, kwargs):
        placement = job_owner.placement[args[0]]
        indicators = placement.get_indicators()
        return f(placement, *args[1:], indicators, **kwargs)

A job has a couple of display variables that can be set: ``_cname`` for the
class name, ``_name`` for the job name and ``_c`` for the chunk. These are used
to display what the workers are doing during parallel execution. This is an experimental
API and subject to sudden change in the future.

"""

import abc
import concurrent.futures
import logging
import pickle
import tempfile
import typing
import warnings
from enum import Enum, auto

from exceptiongroup import ExceptionGroup

from ..exceptions import JobCancelledError, JobPoolError
from . import MPI
from ._util import ErrorModule, MockModule

if typing.TYPE_CHECKING:
    from mpipool import MPIExecutor


class WorkflowError(ExceptionGroup):
    pass


class JobErroredError(Exception):
    def __init__(self, message, error):
        super().__init__(message)
        self.error = error


class JobStatus(Enum):
    # Job has not been queued yet, waiting for dependencies to resolve.
    PENDING = "pending"
    # Job is on the queue.
    QUEUED = "queued"
    # Job is currently running on a worker.
    RUNNING = "running"
    # Job ran successfully.
    SUCCESS = "success"
    # Job failed (an exception was raised).
    FAILED = "failed"
    # Job was cancelled before it started running.
    CANCELLED = "cancelled"
    # Job was killed for some reason.
    ABORTED = "aborted"


class PoolStatus(Enum):
    # Pool Starting
    STARTING = "Starting"
    # Pool Running
    RUNNING = "Running"
    # Pool Ending
    ENDING = "Ending"


class PoolProgressReason(Enum):
    POOL_STATUS_CHANGE = auto()
    JOB_STATUS_CHANGE = auto()
    MAX_TIMEOUT_PING = auto()


class PoolProgress:
    def __init__(self, pool: "JobPool", reason: PoolProgressReason):
        self._pool = pool
        self._reason = reason

    @property
    def reason(self):
        return self._reason

    @property
    def jobs(self):
        return self._pool.jobs

    @property
    def status(self):
        return self._pool.status


class PoolJobUpdateProgress(PoolProgress):
    def __init__(self, pool: "JobPool", job: "Job", old_status: "JobStatus"):
        super().__init__(pool, PoolProgressReason.JOB_STATUS_CHANGE)
        self._job = job
        self._old_status = old_status

    @property
    def job(self):
        return self._job


class PoolStatusProgress(PoolProgress):
    def __init__(self, pool: "JobPool", old_status: PoolStatus):
        super().__init__(pool, PoolProgressReason.POOL_STATUS_CHANGE)
        self._old_status = old_status


class _MissingMPIExecutor(ErrorModule):
    pass


class _MPIPoolModule(MockModule):
    @property
    def MPIExecutor(self) -> typing.Type["MPIExecutor"]:
        return _MissingMPIExecutor(
            "This is not a public interface. Use `.services.JobPool` instead."
        )

    def enable_serde_logging(self):
        import mpipool

        mpipool.enable_serde_logging()


_MPIPool = _MPIPoolModule("mpipool")


def dispatcher(pool_id, job_args):
    job_type, args, kwargs = job_args
    # Get the static job execution handler from this module
    handler = globals()[job_type].execute
    owner = JobPool.get_owner(pool_id)
    # Execute it.
    return handler(owner, args, kwargs)


class SubmissionContext:
    def __init__(self, submitter, chunks=None, **kwargs):
        self._submitter = submitter
        self._chunks = chunks
        self._context = kwargs

    @property
    def name(self):
        if hasattr(self._submitter, "get_node_name"):
            name = self._submitter.get_node_name()
        else:
            name = str(self._submitter)
        return name

    @property
    def chunks(self):
        from ..storage import chunklist

        return chunklist(self._chunks) if self._chunks is not None else None

    def __getattr__(self, key):
        if key in self._context:
            return self._context[key]
        else:
            return self.__getattribute__(key)


class Job(abc.ABC):
    """
    Dispatches the execution of a function through a JobPool
    """

    def __init__(
        self, pool, submission_context: SubmissionContext, args, kwargs, deps=None
    ):
        self.pool_id = pool.id
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._submit_ctx = submission_context
        self._completion_cbs = []
        self._status = JobStatus.PENDING
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = None
        self._res_file = None
        self._error = None

    @property
    def name(self):
        return self._submit_ctx.name

    @property
    def context(self):
        return self._submit_ctx._context

    @property
    def status(self):
        return self._status

    @property
    def result(self):
        try:
            with open(self._res_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            raise JobPoolError(f"Result of {self} is not available.") from None

    @property
    def error(self):
        return self._error

    def serialize(self):
        name = self.__class__.__name__
        # First arg is to find the static `execute` method so that we don't have to
        # serialize any of the job objects themselves but can still use different handlers
        # for different job types.
        return (name, self._args, self._kwargs)

    @staticmethod
    @abc.abstractmethod
    def execute(job_owner, args, kwargs):
        """
        Job handler
        """
        pass

    def on_completion(self, cb):
        self._completion_cbs.append(cb)

    def set_result(self, value):
        dirname = JobPool.get_tmp_folder(self.pool_id)
        try:
            with tempfile.NamedTemporaryFile(
                prefix=dirname + "/", delete=False, mode="wb"
            ) as fp:
                pickle.dump(value, fp)
                self._res_file = fp.name
        except FileNotFoundError as e:
            self.set_exception(e)
        else:
            self.change_status(JobStatus.SUCCESS)

    def set_exception(self, e: Exception):
        self._error = e
        self.change_status(JobStatus.FAILED)

    def _completed(self):
        if self._status != JobStatus.CANCELLED:
            try:
                result = self._future.result()
            except Exception as e:
                self.set_exception(e)
            else:
                self.set_result(result)
        for cb in self._completion_cbs:
            cb(self)

    def _dep_completed(self, dep):
        # Earlier we registered this callback on the completion of our dependencies.
        # When a dep completes we end up here and we discard it as a dependency as it has
        # finished. If the dep returns an error remove the job from the pool, since the dependency have failed.
        self._deps.discard(dep)
        if dep._status is not JobStatus.SUCCESS:
            self.cancel("Job killed for dependency failure")
        else:
            # When all our dependencies have been discarded we can queue ourselves. Unless the
            # pool is serial, then the pool itself just runs all jobs in order.
            if not self._deps and MPI.get_size() > 1:
                # self._pool is set when the pool first tried to enqueue us, but we were still
                # waiting for deps, in the `_enqueue` method below.
                self._enqueue(self._pool)

    def _enqueue(self, pool):
        if not self._deps and self._status is not JobStatus.CANCELLED:
            # Go ahead and submit ourselves to the pool, no dependencies to wait for
            # The dispatcher is run on the remote worker and unpacks the data required
            # to execute the job contents.
            self.change_status(JobStatus.QUEUED)
            self._future = pool._submit(dispatcher, self.pool_id, self.serialize())
        else:
            # We have unfinished dependencies and should wait until we can enqueue
            # ourselves when our dependencies haved all notified us of their completion.
            # Store the reference to the pool though, so later in `_dep_completed` we can
            # call `_enqueue` again ourselves!
            self._pool = pool

    def cancel(self, reason: typing.Optional[str] = None):
        self.change_status(JobStatus.CANCELLED)
        self._error = JobCancelledError() if reason is None else JobCancelledError(reason)
        if self._future:
            if not self._future.cancel():
                warnings.warn(f"Could not cancel {self}, the job is already running.")

    def change_status(self, status: JobStatus):
        old_status = self._status
        self._status = status
        try:
            # Closed pools may have been removed from this map already.
            pool = JobPool._pools[self.pool_id]
        except KeyError:
            pass
        else:
            progress = PoolJobUpdateProgress(pool, self, old_status)
            pool.add_notification(progress)


class PlacementJob(Job):
    """
    Dispatches the execution of a chunk of a placement strategy through a JobPool.
    """

    def __init__(self, pool, strategy, chunk, deps=None):
        args = (strategy.name, chunk)
        context = SubmissionContext(strategy, [chunk])
        super().__init__(pool, context, args, {}, deps=deps)

    @staticmethod
    def execute(job_owner, args, kwargs):
        name, chunk = args
        placement = job_owner.placement[name]
        indicators = placement.get_indicators()
        return placement.place(chunk, indicators, **kwargs)


class ConnectivityJob(Job):
    """
    Dispatches the execution of a chunk of a connectivity strategy through a JobPool.
    """

    def __init__(self, pool, strategy, pre_roi, post_roi, deps=None):
        from bsb.storage import chunklist

        args = (strategy.name, pre_roi, post_roi)
        context = SubmissionContext(
            strategy, chunks=chunklist((*(pre_roi or []), *(post_roi or [])))
        )
        super().__init__(pool, context, args, {}, deps=deps)

    @staticmethod
    def execute(job_owner, args, kwargs):
        name = args[0]
        connectivity = job_owner.connectivity[name]
        collections = connectivity._get_connect_args_from_job(*args[1:])
        return connectivity.connect(*collections, **kwargs)


class FunctionJob(Job):
    def __init__(self, pool, f, args, kwargs, deps=None, submitter={}):
        self._f = f
        new_args = [f]
        new_args.extend(args)
        context = SubmissionContext(f, chunks=new_args, **submitter)
        super().__init__(pool, context, new_args, kwargs, deps=deps)

    @staticmethod
    def execute(job_owner, args, kwargs):
        f = args[0]
        result = f(job_owner, *args[1:], **kwargs)
        return result


class JobPool:
    _next_pool_id = 0
    _pools = {}
    _pool_owners = {}
    _tmp_folders = {}

    def __init__(self, scaffold, fail_fast=False):
        self._running_futures: list[concurrent.futures.Future] = []
        self._pool: typing.Optional["MPIExecutor"] = None
        self._job_queue: list[Job] = []
        self.id = JobPool._next_pool_id
        self._listeners = []
        self._max_wait = 60
        self._status = PoolStatus.STARTING
        self._progress_notifications: list["PoolProgress"] = []
        self._workers_raise_unhandled = False
        self._fail_fast = fail_fast
        JobPool._next_pool_id += 1
        JobPool._pool_owners[self.id] = scaffold
        JobPool._pools[self.id] = self

    def add_listener(self, listener, max_wait=None):
        self._max_wait = min(self._max_wait, max_wait or float("+inf"))
        self._listeners.append(listener)

    @property
    def status(self):
        return self._status

    @property
    def jobs(self):
        return [*self._job_queue]

    @property
    def parallel(self):
        return MPI.get_size() > 1

    @classmethod
    def get_owner(cls, id):
        return cls._pool_owners[id]

    @classmethod
    def get_tmp_folder(cls, id):
        return cls._tmp_folders[id]

    @property
    def owner(self):
        return self.get_owner(self.id)

    def is_main(self):
        return MPI.get_rank() == 0

    def _put(self, job):
        """
        Puts a job onto our internal queue.
        """
        if self._pool and not self._pool.open:
            raise JobPoolError("No job pool available for job submission.")
        else:
            self._job_queue.append(job)

    def _submit(self, fn, *args, **kwargs):
        if not self._pool or not self._pool.open:
            raise JobPoolError("No job pool available for job submission.")
        else:
            future = self._pool.submit(fn, *args, **kwargs)
            self._running_futures.append(future)
            return future

    def queue(self, f, args=None, kwargs=None, deps=None, submitter={}):
        job = FunctionJob(self, f, args or [], kwargs or {}, deps, submitter=submitter)
        self._put(job)
        return job

    def queue_placement(self, strategy, chunk, deps=None):
        job = PlacementJob(self, strategy, chunk, deps)
        self._put(job)
        return job

    def queue_connectivity(self, strategy, pre_roi, post_roi, deps=None):
        job = ConnectivityJob(self, strategy, pre_roi, post_roi, deps)
        self._put(job)
        return job

    def execute(self, return_results=False):
        """
        Execute the jobs in the queue

        In serial execution this runs all of the jobs in the queue in First In First Out
        order. In parallel execution this enqueues all jobs into the MPIPool unless they
        have dependencies that need to complete first.
        """

        try:
            with tempfile.TemporaryDirectory() as tmp_dirname:
                JobPool._tmp_folders[self.id] = tmp_dirname
                if self.parallel:
                    self._execute_parallel()
                else:
                    self._execute_serial()

                # fixme: if you unindent this by 1, segmentation faults occur
                if return_results:
                    return {
                        job: job.result
                        for job in self._job_queue
                        if job.status == JobStatus.SUCCESS
                    }
        finally:
            # Clean up pool/job references
            self._job_queue = []
            del JobPool._pools[self.id]

    def _execute_parallel(self):
        import bsb.options

        if bsb.options.debug_pool:
            _MPIPool.enable_serde_logging()
        # Create the MPI pool
        self._pool = _MPIPool.MPIExecutor(
            loglevel=logging.DEBUG if bsb.options.debug_pool else logging.CRITICAL
        )

        if self._pool.is_worker():
            # The workers will return out of the pool constructor when they receive
            # the shutdown signal from the master, they return here skipping the
            # master logic.

            # Check if we need to abort our process due to errors etc.
            abort = MPI.bcast(None)
            if abort:
                raise WorkflowError(
                    "Unhandled exceptions during parallel execution.",
                    [RuntimeError("See main node logs for details.")],
                )
            return

        unhandled = []
        try:
            # Tell each job in our queue that they have to put themselves in the pool
            # queue; each job will store their own future and will use the futures of
            # their previously enqueued dependencies to determine when they can put
            # themselves on the pool queue.
            for job in self._job_queue:
                job._enqueue(self)

            # Tell the listeners execution is running
            self.change_status(PoolStatus.RUNNING)

            # As long as any of the jobs aren't done yet we repeat the wait action with a timeout
            while any(
                job.status == JobStatus.PENDING or job.status == JobStatus.QUEUED
                for job in self._job_queue
            ):
                done, not_done = concurrent.futures.wait(
                    self._running_futures,
                    timeout=self._max_wait,
                    return_when="FIRST_COMPLETED",
                )
                # Complete any jobs that are done
                for job in self._job_queue:
                    if job._future in done:
                        job._completed()
                # Remove running futures that are done
                for future in done:
                    self._running_futures.remove(future)
                # If nothing finished, post a timeout notification.
                if not len(done):
                    self.add_notification(
                        PoolProgress(self, PoolProgressReason.MAX_TIMEOUT_PING)
                    )
                # Notify all the listeners, and store/raise any unhandled errors
                unhandled.extend(self.notify())
                if unhandled and self._fail_fast:
                    self.raise_unhandled(unhandled)

            # Notify listeners that execution is over
            self.change_status(PoolStatus.ENDING)
            # Raise any unhandled errors
            if unhandled:
                self.raise_unhandled(unhandled)
        except:
            # If any exception (including SystemExit and KeyboardInterrupt) happen on main, we should
            # broadcast the abort to all worker nodes.
            self._workers_raise_unhandled = True
            raise
        finally:
            # Shut down our internal pool
            self._pool.shutdown(wait=False, cancel_futures=True)
            # Broadcast whether the worker nodes should raise an unhandled error.
            MPI.bcast(self._workers_raise_unhandled)

    def _execute_serial(self):
        # Prepare jobs for local execution
        for job in self._job_queue:
            job._future = concurrent.futures.Future()
            job._status = JobStatus.QUEUED

        self.change_status(PoolStatus.RUNNING)

        unhandled = []
        # Just run each job serially
        for job in self._job_queue:
            f = job._future
            try:
                job._status = JobStatus.RUNNING
                # Execute the static handler
                result = job.execute(self.owner, job._args, job._kwargs)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(result)
            finally:
                f.done()
            job._completed()
            unhandled.extend(self.notify())
            if unhandled and self._fail_fast:
                self.raise_unhandled(unhandled)
        if unhandled:
            self.raise_unhandled(unhandled)

        self.change_status(PoolStatus.ENDING)

    def change_status(self, status: PoolStatus):
        old_status = self._status
        self._status = status
        self.add_notification(PoolStatusProgress(self, old_status))
        self.notify()

    def add_notification(self, notification: PoolProgress):
        self._progress_notifications.append(notification)

    def notify(self):
        unhandled_errors = []
        for notification in self._progress_notifications:
            job = getattr(notification, "job", None)
            has_error = getattr(job, "error", None) is not None
            handled_error = [bool(listener(notification)) for listener in self._listeners]
            if has_error and not any(handled_error):
                unhandled_errors.append(job)
        self._progress_notifications = []
        return unhandled_errors

    def raise_unhandled(self, unhandled):
        errors = []
        # Raise and catch for nicer traceback
        for job in unhandled:
            try:
                raise JobErroredError(f"{job.name} job failed", job.error) from job.error
            except JobErroredError as e:
                errors.append(e)
        raise WorkflowError(
            f"Your workflow encountered errors.",
            errors,
        )
