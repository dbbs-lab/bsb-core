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
import typing
import warnings
from enum import Enum
from time import sleep

import numpy as np

from ..exceptions import JobCancelledError
from . import MPI
from ._util import ErrorModule, MockModule

if typing.TYPE_CHECKING:
    from mpipool import MPIExecutor


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


class _MissingMPIExecutor(ErrorModule):
    pass


class _MPIPoolModule(MockModule):
    @property
    def MPIExecutor(self):
        return _MissingMPIExecutor(
            "This is not a public interface. Use `.services.JobPool` instead."
        )


_MPIPool = _MPIPoolModule("mpipool")


def dispatcher(pool_id, job_args):
    job_type, args, kwargs = job_args
    # Get the static job execution handler from this module
    handler = globals()[job_type].execute
    owner = JobPool.get_owner(pool_id)
    # Execute it.
    return handler(owner, args, kwargs)


class Job(abc.ABC):
    """
    Dispatches the execution of a function through a JobPool
    """

    def __init__(self, pool, args, kwargs, deps=None, submitter=None):
        self.pool_id = pool.id
        self._name = "No name"
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._submitter = submitter
        self._completion_cbs = []
        self._status = JobStatus.PENDING
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = None
        self._result = None
        self._error = None

    @property
    def status(self):
        return self._status

    @property
    def result(self):
        return self._result

    @property
    def error(self):
        return self._error

    def serialize(self):
        name = self.__class__.__name__
        # First arg is to find the static `execute` method so that we don't have to
        # serialize any of the job objects themselves but can still use different handlers
        # for different job types.
        return (name, self._args, self._kwargs)

    @abc.abstractmethod
    def execute(job_owner, args, kwargs):
        """
        Job handler
        """
        pass

    def on_completion(self, cb):
        self._completion_cbs.append(cb)

    def _completion(self, future):
        # todo: First update ourselves, based on future:
        #       * retrieve error/result
        #       then, publish ourselves to all our listeners:
        if self._status != JobStatus.CANCELLED:
            try:
                self._result = future.result()
            except Exception as e:
                self._status = JobStatus.FAILED
                self._error = e
            else:
                self._status = JobStatus.SUCCESS
        for cb in self._completion_cbs:
            cb(self, err=None, result=None)

    def _dep_completed(self, dep, err=None, result=None):
        # todo: use err/result
        #       * todo maybe: write result to file, then read file for data in dependency submit
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
            self._status = JobStatus.QUEUED
            self._future = pool._submit(dispatcher, self.pool_id, self.serialize())
            # Invoke our completion callbacks when the future completes.
            self._future.add_done_callback(self._completion)
        else:
            # We have unfinished dependencies and should wait until we can enqueue
            # ourselves when our dependencies haved all notified us of their completion.
            # Store the reference to the pool though, so later in `_dep_completed` we can
            # call `_enqueue` again ourselves!
            self._pool = pool

    def cancel(self, reason: typing.Optional[str] = None):
        self._status = JobStatus.CANCELLED
        self._error = JobCancelledError() if reason is None else JobCancelledError(reason)
        if self._future:
            if not self._future.cancel():
                warnings.warn(f"Could not cancel {self}, the job is already running.")


class PlacementJob(Job):
    """
    Dispatches the execution of a chunk of a placement strategy through a JobPool.
    """

    def __init__(self, pool, strategy, chunk, deps=None):
        args = (strategy.name, chunk)
        super().__init__(pool, args, {}, deps=deps)
        self._cname = strategy.__class__.__name__
        self._name = strategy.name

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
        args = (strategy.name, pre_roi, post_roi)
        super().__init__(pool, args, {}, deps=deps)
        self._cname = strategy.__class__.__name__
        self._name = strategy.name

    @staticmethod
    def execute(job_owner, args, kwargs):
        name = args[0]
        connectivity = job_owner.connectivity[name]
        collections = connectivity._get_connect_args_from_job(*args[1:])
        return connectivity.connect(*collections, **kwargs)


class FunctionJob(Job):
    def __init__(self, pool, f, args, kwargs, deps=None, submitter=None):
        self._f = f
        self._name = f.__name__
        new_args = [f]
        new_args.extend(args)
        super().__init__(pool, new_args, kwargs, deps=deps, submitter=submitter)

    @staticmethod
    def execute(job_owner, args, kwargs):
        f = args[0]
        result = f(job_owner, *args[1:], **kwargs)
        return result


class JobPool:
    _next_pool_id = 0
    _pool_owners = {}

    def __init__(self, scaffold):
        self._running_futures: list[concurrent.futures.Future] = []
        self._pool: typing.Optional["MPIExecutor"] = None
        self._job_queue: list[Job] = []
        self.id = JobPool._next_pool_id
        self._listeners = []
        self._max_wait = 60
        self._status = PoolStatus.STARTING
        JobPool._next_pool_id += 1
        JobPool._pool_owners[self.id] = scaffold

    def add_listener(self, listener, max_wait=None):
        self._max_wait = min(self._max_wait, max_wait or float("+inf"))
        self._listeners.append(listener)

    @property
    def parallel(self):
        return MPI.get_size() > 1

    @classmethod
    def get_owner(cls, id):
        return cls._pool_owners[id]

    @property
    def owner(self):
        return self.get_owner(self.id)

    def is_master(self):
        return MPI.get_rank() == 0

    def _put(self, job):
        """
        Puts a job onto our internal queue. Putting items in the queue does not mean they
        will be executed. For this ``self.execute()`` must be called.
        """
        # This complements the event loop in parallel execution and is
        # executed synchronously in serial execution.
        self._job_queue.append(job)

    def _submit(self, fn, *args, **kwargs):
        if self._pool and self._pool.open:
            future = self._pool.submit(fn, *args, **kwargs)
            future.add_done_callback(lambda me: self._running_futures.remove(me))
            self._running_futures.append(future)
            return future
        else:
            raise RuntimeError("Attempting to submit job to closed pool.")

    def _job_cancel(self, job, msg="Bo"):
        job.cancel(msg)

    def queue(self, f, args=None, kwargs=None, deps=None, submitter=None):
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

    def execute(self):
        """
        Execute the jobs in the queue

        In serial execution this runs all of the jobs in the queue in First In First Out
        order. In parallel execution this enqueues all jobs into the MPIPool unless they
        have dependencies that need to complete first.
        """
        if self.parallel:
            # Create the MPI pool
            self._pool = _MPIPool.MPIExecutor()

            if self._pool.is_worker():
                # The workers will return out of the pool constructor when they receive
                # the shutdown signal from the master, they return here skipping the
                # master logic.
                return

            try:
                # Tell each job in our queue that they have to put themselves in the pool
                # queue; each job will store their own future and will use the futures of
                # their previously enqueued dependencies to determine when they can put
                # themselves on the pool queue.

                for job in self._job_queue:
                    job._enqueue(self)

                # Call the listeners when execution is starting
                for listener in self._listeners:
                    listener(self._job_queue, self._status)
                # Now we start to listen to future, use the boolean check_failures variable to exit when an error is raised
                self._status = PoolStatus.RUNNING
                self._check_failures = False
                # As long as any of the jobs aren't done yet we repeat the wait action with a timeout defined by _max_wait
                while any(
                    [
                        job._status == JobStatus.PENDING
                        or job._status == JobStatus.QUEUED
                        for job in self._job_queue
                    ]
                ):
                    concurrent.futures.wait(
                        self._running_futures,
                        timeout=self._max_wait,
                        return_when="FIRST_COMPLETED",
                    )
                    # Send the updates to the listeners and check if an error is raised
                    for listener in self._listeners:
                        listener(self._job_queue, self._status)
            finally:
                self._status = PoolStatus.ENDING
                for listener in self._listeners:
                    listener(self._job_queue, self._status)
                self._pool.shutdown()

        else:
            # todo: start a thread/coroutines/asyncio that calls the listeners periodically
            # Prepare jobs for local execution
            for job in self._job_queue:
                job._future = concurrent.futures.Future()
                job._future.add_done_callback(job._completion)
                job._status = JobStatus.QUEUED

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
                for listener in self._listeners:
                    listener(self._job_queue, self._status)

        # Clear the queue after all jobs have been done
        self._job_queue = []
