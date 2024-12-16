"""
Job pooling module.

Jobs derive from the base :class:`.Job` class which can be put on the queue of a
:class:`.JobPool`. In order to submit themselves to the pool Jobs will
:meth:`~.Job.serialize` themselves into a predefined set of variables::

   job.serialize() -> (job_type, f, args, kwargs)

* ``job_type`` should be a string that is a class name defined in this module.
   (e.g. ``"PlacementJob"``)

* ``f`` should be the function object that the job's ``execute`` method should
   execute.

* ``args`` and ``kwargs`` are the args to be passed to that ``f``.

The :meth:`.Job.execute` handler can help interpret ``args`` and ``kwargs``
before running ``f``. The execute handler has access to the scaffold on the MPI
process so one best serializes just the name of some part of the configuration,
rather than trying to pickle the complex objects. For example, the
:class:`.PlacementJob` uses the first ``args`` element to store the
:class:`~bsb.placement.strategy.PlacementStrategy` name and then retrieve it from the
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
import functools
import logging
import pickle
import tempfile
import threading
import typing
import warnings
import zlib
from contextlib import ExitStack
from enum import Enum, auto

import numpy as np
from exceptiongroup import ExceptionGroup

from .._util import obj_str_insert
from ..exceptions import (
    JobCancelledError,
    JobPoolContextError,
    JobPoolError,
    JobSchedulingError,
)
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
    # Pool has been initialized and jobs can be scheduled.
    SCHEDULING = "scheduling"
    # Pool started execution.
    EXECUTING = "executing"
    # Pool is closing down.
    CLOSING = "closing"


class PoolProgressReason(Enum):
    POOL_STATUS_CHANGE = auto()
    JOB_ADDED = auto()
    JOB_STATUS_CHANGE = auto()
    MAX_TIMEOUT_PING = auto()


class Workflow:
    def __init__(self, phases: list[str]):
        self._phases = phases
        self._phase = 0

    @property
    def phases(self):
        return [*self._phases]

    @property
    def finished(self):
        return self._phase >= len(self._phases)

    @property
    def phase(self):
        if self.finished:
            return "finished"
        else:
            return self._phases[self._phase]

    def next_phase(self):
        self._phase += 1
        return self.phase


class PoolProgress:
    """
    Class used to report pool progression to listeners.
    """

    def __init__(self, pool: "JobPool", reason: PoolProgressReason):
        self._pool = pool
        self._reason = reason

    @property
    def reason(self):
        return self._reason

    @property
    def workflow(self):
        return self._pool.workflow

    @property
    def jobs(self):
        return self._pool.jobs

    @property
    def status(self):
        return self._pool.status


class PoolJobAddedProgress(PoolProgress):
    def __init__(self, pool: "JobPool", job: "Job"):
        super().__init__(pool, PoolProgressReason.JOB_ADDED)
        self._job = job

    @property
    def job(self):
        return self._job


class PoolJobUpdateProgress(PoolProgress):
    def __init__(self, pool: "JobPool", job: "Job", old_status: "JobStatus"):
        super().__init__(pool, PoolProgressReason.JOB_STATUS_CHANGE)
        self._job = job
        self._old_status = old_status

    @property
    def job(self):
        return self._job

    @property
    def old_status(self):
        return self._old_status

    @property
    def status(self):
        return self._job.status


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
    """
    The dispatcher is the function that gets pickled on main, and unpacked "here" on the
    worker. Through class variables on `JobPool` and the given `pool_id` we can find the
    pool and scaffold object, and the job function to run.

    Before running a job, the cache is checked for eventual cached items to free up.
    """
    job_type, args, kwargs = job_args
    # Get the static job execution handler from this module
    handler = globals()[job_type].execute
    # Get the owning scaffold from the JobPool class variables, which act as a registry.
    owner = JobPool.get_owner(pool_id)

    # Check the pool's cache
    pool = JobPool._pools[pool_id]
    required_cache_items = pool._read_required_cache_items()
    # and free any stale cached items
    free_stale_pool_cache(owner, required_cache_items)

    # Execute the job handler.
    return handler(owner, args, kwargs)


class SubmissionContext:
    """
    Context information on who submitted a certain job.
    """

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
    def submitter(self):
        return self._submitter

    @property
    def chunks(self):
        from ..storage._chunks import chunklist

        return chunklist(self._chunks) if self._chunks is not None else None

    @property
    def context(self):
        return {**self._context}

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
        self,
        pool,
        submission_context: SubmissionContext,
        args,
        kwargs,
        deps=None,
        cache_items=None,
    ):
        self.pool_id = pool.id
        self._comm = pool._comm
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._submit_ctx = submission_context
        self._completion_cbs = []
        self._status = JobStatus.PENDING
        self._future: typing.Optional[concurrent.futures.Future] = None
        self._thread: typing.Optional[threading.Thread] = None
        self._res_file = None
        self._error = None
        self._cache_items: list[int] = [] if cache_items is None else cache_items

        for j in self._deps:
            j.on_completion(self._dep_completed)

    @obj_str_insert
    def __str__(self):
        return self.description

    @property
    def name(self):
        return self._submit_ctx.name

    @property
    def description(self):
        descr = self.name
        if self.context:
            descr += " (" + ", ".join(f"{k}={v}" for k, v in self.context.items()) + ")"
        return descr

    @property
    def submitter(self):
        return self._submit_ctx.submitter

    @property
    def context(self):
        return self._submit_ctx.context

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
        """
        Convert the job to a (de)serializable representation
        """
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

    def run(self, timeout=None):
        """
        Execute the job on the current process, in a thread, and return whether the job is still running.
        """
        if self._thread is None:

            def target():
                try:
                    # Execute the static handler
                    result = self.execute(self._pool.owner, self._args, self._kwargs)
                except Exception as e:
                    self._future.set_exception(e)
                else:
                    self._future.set_result(result)

            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._completed()
            return False
        return True

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
            if not self._deps and self._comm.get_size() > 1:
                # self._pool is set when the pool first tried to enqueue us, but we were still
                # waiting for deps, in the `_enqueue` method below.
                self._enqueue(self._pool)

    def _enqueue(self, pool):
        if not self._deps and self._status is JobStatus.PENDING:
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
        cache_items = get_node_cache_items(strategy)
        super().__init__(pool, context, args, {}, deps=deps, cache_items=cache_items)

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
        from ..storage._chunks import chunklist

        args = (strategy.name, pre_roi, post_roi)
        context = SubmissionContext(
            strategy, chunks=chunklist((*(pre_roi or []), *(post_roi or [])))
        )
        cache_items = get_node_cache_items(strategy)
        super().__init__(pool, context, args, {}, deps=deps, cache_items=cache_items)

    @staticmethod
    def execute(job_owner, args, kwargs):
        name = args[0]
        connectivity = job_owner.connectivity[name]
        collections = connectivity._get_connect_args_from_job(*args[1:])
        return connectivity.connect(*collections, **kwargs)


class FunctionJob(Job):
    def __init__(self, pool, f, args, kwargs, deps=None, cache_items=None, **context):
        # Pack the function into the args
        args = (f, args)
        # If no submitter was given, set the function as submitter
        context.setdefault("submitter", f)
        super().__init__(
            pool,
            SubmissionContext(**context),
            args,
            kwargs,
            deps=deps,
            cache_items=cache_items,
        )

    @staticmethod
    def execute(job_owner, args, kwargs):
        # Unpack the function from the args
        f, args = args
        return f(job_owner, *args, **kwargs)


class JobPool:
    _next_pool_id = 0
    _pools = {}
    _pool_owners = {}
    _tmp_folders = {}

    def __init__(self, id, scaffold, fail_fast=False, workflow: "Workflow" = None):
        self._schedulers: list[concurrent.futures.Future] = []
        self.id: int = id
        self._scaffold = scaffold
        self._comm = scaffold._comm
        self._unhandled_errors = []
        self._running_futures: list[concurrent.futures.Future] = []
        self._mpipool: typing.Optional["MPIExecutor"] = None
        self._job_queue: list[Job] = []
        self._listeners = []
        self._max_wait = 60
        self._status: PoolStatus = None
        self._progress_notifications: list["PoolProgress"] = []
        self._workers_raise_unhandled = False
        self._fail_fast = fail_fast
        self._workflow = workflow
        self._cache_buffer = np.zeros(1000, dtype=np.uint64)
        self._cache_window = self._comm.window(self._cache_buffer)

    def __enter__(self):
        self._context = ExitStack()
        tmp_dirname = self._context.enter_context(tempfile.TemporaryDirectory())

        JobPool._pool_owners[self.id] = self._scaffold
        JobPool._pools[self.id] = self
        JobPool._tmp_folders[self.id] = tmp_dirname
        del self._scaffold

        for listener in self._listeners:
            try:
                self._context.enter_context(listener)
            except (TypeError, AttributeError):
                # Listener is not a context manager
                pass
        self.change_status(PoolStatus.SCHEDULING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)
        # Clean up pool/job references
        self._job_queue = []
        del JobPool._pools[self.id]
        del JobPool._pool_owners[self.id]
        del JobPool._tmp_folders[self.id]
        self.id = None

    def add_listener(self, listener, max_wait=None):
        self._max_wait = min(self._max_wait, max_wait or float("+inf"))
        self._listeners.append(listener)

    @property
    def workflow(self):
        return self._workflow

    @property
    def status(self):
        return self._status

    @property
    def jobs(self) -> list[Job]:
        return [*self._job_queue]

    @property
    def parallel(self):
        return self._comm.get_size() > 1

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
        return self._comm.get_rank() == 0

    def get_submissions_of(self, submitter):
        return [job for job in self._job_queue if job.submitter is submitter]

    def _put(self, job):
        """
        Puts a job onto our internal queue.
        """
        if self._mpipool and not self._mpipool.open:
            raise JobPoolError("No job pool available for job submission.")
        else:
            self.add_notification(PoolJobAddedProgress(self, job))
            self._job_queue.append(job)
            if self._mpipool:
                # This job was scheduled after the MPIPool was opened, so immediately
                # put it on the MPIPool's queue.
                job._enqueue(self)

    def _submit(self, fn, *args, **kwargs):
        if not self._mpipool or not self._mpipool.open:
            raise JobPoolError("No job pool available for job submission.")
        else:
            future = self._mpipool.submit(fn, *args, **kwargs)
            self._running_futures.append(future)
            return future

    def _schedule(self, future: concurrent.futures.Future, nodes, scheduler):
        _failed_nodes = []
        if not future.set_running_or_notify_cancel():
            return
        try:
            for node in nodes:
                failed_deps = [
                    n for n in getattr(node, "depends_on", []) if n in _failed_nodes
                ]
                if failed_deps:
                    _failed_nodes.append(node)
                    ctx = SubmissionContext(
                        node,
                        error=JobSchedulingError(
                            f"Depends on {failed_deps}, whom failed."
                        ),
                    )
                    self._unhandled_errors.append(ctx)
                    continue
                try:
                    scheduler(node)
                except Exception as e:
                    _failed_nodes.append(node)
                    ctx = SubmissionContext(node, error=e)
                    self._unhandled_errors.append(ctx)
        finally:
            future.set_result(None)

    def schedule(self, nodes, scheduler=None):
        if scheduler is None:

            def scheduler(node):
                node.queue(self)

        future = concurrent.futures.Future()
        self._schedulers.append(future)
        thread = threading.Thread(target=self._schedule, args=(future, nodes, scheduler))
        thread.start()

    @property
    def scheduling(self):
        return any(not f.done() for f in self._schedulers)

    def queue(self, f, args=None, kwargs=None, deps=None, **context):
        job = FunctionJob(self, f, args or [], kwargs or {}, deps, [], **context)
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

        In serial execution this runs all the jobs in the queue in First In First Out
        order. In parallel execution this enqueues all jobs into the MPIPool unless they
        have dependencies that need to complete first.
        """

        if self.id is None:
            raise JobPoolContextError("Job pools must use a context manager.")

        if self.parallel:
            self._execute_parallel()
        else:
            self._execute_serial()

        if return_results:
            return {
                job: job.result
                for job in self._job_queue
                if job.status == JobStatus.SUCCESS
            }

    def _execute_parallel(self):
        import bsb.options

        # Enable full mpipool debugging
        if bsb.options.debug_pool:
            _MPIPool.enable_serde_logging()
        # Create the MPI pool
        self._mpipool = _MPIPool.MPIExecutor(
            loglevel=logging.DEBUG if bsb.options.debug_pool else logging.CRITICAL
        )

        if self._mpipool.is_worker():
            # The workers will return out of the pool constructor when they receive
            # the shutdown signal from the master, they return here skipping the
            # master logic.

            # Check if we need to abort our process due to errors etc.
            abort = self._comm.bcast(None)
            if abort:
                raise WorkflowError(
                    "Unhandled exceptions during parallel execution.",
                    [JobPoolError("See main node logs for details.")],
                )

            # Free all cached items
            free_stale_pool_cache(self.owner, set())

            return

        try:
            # Tell the listeners execution is running
            self.change_status(PoolStatus.EXECUTING)
            # Kickstart the workers with the queued jobs
            for job in self._job_queue:
                job._enqueue(self)
            # Add the scheduling futures to the running futures, to await them.
            self._running_futures.extend(self._schedulers)
            # Start tracking cached items
            self._update_cache_window()

            # Keep executing as long as any of the schedulers or jobs aren't done yet.
            while self.scheduling or any(
                job.status == JobStatus.PENDING or job.status == JobStatus.QUEUED
                for job in self._job_queue
            ):
                try:
                    done, not_done = concurrent.futures.wait(
                        self._running_futures,
                        timeout=self._max_wait,
                        return_when="FIRST_COMPLETED",
                    )
                except ValueError:
                    # Sometimes a ValueError is raised here, perhaps because we modify
                    # the list below?
                    continue

                # Complete any jobs that are done
                for job in self._job_queue:
                    if job._future in done:
                        job._completed()

                # If a job finished, update the required cache items
                if len(done):
                    self._update_cache_window()

                # Remove running futures that are done
                for future in done:
                    self._running_futures.remove(future)
                # If nothing finished, post a timeout notification.
                if not len(done):
                    self.ping()
                # Notify all the listeners, and store/raise any unhandled errors
                self.notify()

            # Notify listeners that execution is over
            self.change_status(PoolStatus.CLOSING)
            # Raise any unhandled errors
            self.raise_unhandled()
        except:
            # If any exception (including SystemExit and KeyboardInterrupt) happen on main, we should
            # broadcast the abort to all worker nodes.
            self._workers_raise_unhandled = True
            raise
        finally:
            # Shut down our internal pool
            self._mpipool.shutdown(wait=False, cancel_futures=True)
            # Broadcast whether the worker nodes should raise an unhandled error.
            self._comm.bcast(self._workers_raise_unhandled)

    def _execute_serial(self):
        # Wait for jobs to finish scheduling
        while concurrent.futures.wait(
            self._schedulers, timeout=self._max_wait, return_when="FIRST_COMPLETED"
        )[1]:
            self.ping()
            self.notify()
        # Prepare jobs for local execution
        for job in self._job_queue:
            job._future = concurrent.futures.Future()
            job._pool = self
            if job.status != JobStatus.CANCELLED and job.status != JobStatus.ABORTED:
                job._status = JobStatus.QUEUED
            else:
                job._future.cancel()

        self.change_status(PoolStatus.EXECUTING)
        # Just run each job serially
        for job in self._job_queue:
            if not job._future.set_running_or_notify_cancel():
                continue
            job.change_status(JobStatus.RUNNING)
            self.notify()
            while job.run(timeout=self._max_wait):
                self.ping()
                self.notify()
            # After each job, check if any cache items can be freed.
            free_stale_pool_cache(self.owner, self.get_required_cache_items())
            self.notify()
        # Raise any unhandled errors
        self.raise_unhandled()

        self.change_status(PoolStatus.CLOSING)

    def change_status(self, status: PoolStatus):
        old_status = self._status
        self._status = status
        self.add_notification(PoolStatusProgress(self, old_status))
        self.notify()

    def add_notification(self, notification: PoolProgress):
        self._progress_notifications.append(notification)

    def ping(self):
        self.add_notification(PoolProgress(self, PoolProgressReason.MAX_TIMEOUT_PING))

    def notify(self):
        for notification in self._progress_notifications:
            job = getattr(notification, "job", None)
            job_error = getattr(job, "error", None)
            has_error = job_error is not None and type(job_error) is not JobCancelledError
            handled_error = [bool(listener(notification)) for listener in self._listeners]
            if has_error and not any(handled_error):
                self._unhandled_errors.append(job)
        if self._fail_fast:
            self.raise_unhandled()
        self._progress_notifications = []

    def raise_unhandled(self):
        if not self._unhandled_errors:
            return
        errors = []
        # Raise and catch for nicer traceback
        for job in self._unhandled_errors:
            try:
                if isinstance(job, SubmissionContext):
                    raise JobSchedulingError(
                        f"{job.name} failed to schedule its jobs."
                    ) from job.context["error"]
                raise JobErroredError(f"{job} failed", job.error) from job.error
            except (JobErroredError, JobSchedulingError) as e:
                errors.append(e)
        self._unhandled_errors = []
        raise WorkflowError(
            f"Your workflow encountered errors.",
            errors,
        )

    def get_required_cache_items(self):
        """
        Returns the list of cache functions for all the jobs in the queue

        :return: set of cache function name
        :rtype: set[int]
        """
        items = set()
        for job in self._job_queue:
            if (
                job.status == JobStatus.QUEUED
                or job.status == JobStatus.PENDING
                or job.status == JobStatus.RUNNING
            ):
                items.update(job._cache_items)
        return items

    def _update_cache_window(self):
        """
        Checks and updates if the cache buffer should be updated by looking at the job
        statuses in the job queue. Only call on main.
        """
        # Create a new cache window buffer
        new_buffer = np.zeros(1000, dtype=int)
        for i, item in enumerate(self.get_required_cache_items()):
            new_buffer[i] = item

        # If there are actual cache requirement differences, lock the window
        # and transfer the buffer
        if np.any(new_buffer != self._cache_buffer):
            self._cache_window.Lock(0)
            self._cache_buffer[:] = new_buffer
            self._cache_window.Unlock(0)

    def _read_required_cache_items(self):
        """
        Locks the cache window and read the still required cache items from rank 0.
        Only call on workers.
        """
        from mpi4py.MPI import UINT64_T

        self._cache_window.Lock(0)
        self._cache_window.Get([self._cache_buffer, UINT64_T], 0)
        self._cache_window.Unlock(0)
        return set(self._cache_buffer)


def get_node_cache_items(node):
    return [
        attr.get_pool_cache_id(node)
        for key in dir(node)
        if hasattr(attr := getattr(node, key), "get_pool_cache_id")
    ]


def free_stale_pool_cache(scaffold, required_cache_items: set[int]):
    for stale_key in set(scaffold._pool_cache.keys()) - required_cache_items:
        # If so, pop them and execute the registered cleanup function.
        scaffold._pool_cache.pop(stale_key)()


def pool_cache(caching_function):
    @functools.cache
    def decorated(self, *args, **kwargs):
        self.scaffold.register_pool_cached_item(
            decorated.get_pool_cache_id(self), cleanup
        )
        return caching_function(self, *args, **kwargs)

    def get_pool_cache_id(node):
        if not hasattr(node, "get_node_name"):
            raise RuntimeError(
                "Pool caching can only be used on methods of @node decorated classes."
            )
        return _cache_hash(f"{node.get_node_name()}.{caching_function.__name__}")

    def cleanup():
        decorated.cache_clear()

    decorated.get_pool_cache_id = get_pool_cache_id

    return decorated


def _cache_hash(string):
    return zlib.crc32(string.encode())
