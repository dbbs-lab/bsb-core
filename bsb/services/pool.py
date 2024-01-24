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
    handler(owner, args, kwargs)


class Job(abc.ABC):
    """
    Dispatches the execution of a function through a JobPool
    """

    def __init__(self, pool, args, kwargs, deps=None, submitter=None):
        self.pool_id = pool.id
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._completion_cbs = []
        self._status = JobStatus.PENDING
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = None
        self._result = None
        self._error = None

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
        print(f" Job {dep} discarded, status: {dep._status}")
        if dep._status is not JobStatus.SUCCESS:
            self.cancel()
        else:
            # When all our dependencies have been discarded we can queue ourselves. Unless the
            # pool is serial, then the pool itself just runs all jobs in order.
            print(f"Dependecies: {not self._deps}. ")
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
                warnings.warn(f"Could not cancel {self}")


class PlacementJob(Job):
    """
    Dispatches the execution of a chunk of a placement strategy through a JobPool.
    """

    def __init__(self, pool, strategy, chunk, deps=None):
        self._f = strategy.place.__func__
        args = (self._f, strategy.name, chunk)
        super().__init__(pool, args, {}, deps=deps)
        self._cname = strategy.__class__.__name__
        self._name = strategy.name
        self._c = chunk

    @staticmethod
    def execute(job_owner, args, kwargs):
        name = args[1]
        f = args[0]
        placement = job_owner.placement[name]
        indicators = placement.get_indicators()
        return f(placement, *args[2:], indicators, **kwargs)


class ConnectivityJob(Job):
    """
    Dispatches the execution of a chunk of a connectivity strategy through a JobPool.
    """

    def __init__(self, pool, strategy, pre_roi, post_roi, deps=None):
        self._f = strategy.connect.__func__
        args = (self._f, strategy.name, pre_roi, post_roi)
        super().__init__(pool, args, {}, deps=deps)
        self._cname = strategy.__class__.__name__
        self._name = strategy.name

    @staticmethod
    def execute(job_owner, args, kwargs):
        name = args[1]
        f = args[0]
        connectivity = job_owner.connectivity[name]
        collections = connectivity._get_connect_args_from_job(*args[2:])
        return f(connectivity, *collections, **kwargs)


class FunctionJob(Job):
    def __init__(self, pool, f, args, kwargs, deps=None):
        self._f = f
        new_args = [f]
        new_args.extend(args)
        super().__init__(pool, new_args, kwargs, deps=deps)

    @staticmethod
    def execute(job_owner, args, kwargs):
        f = args[0]
        return f(job_owner, *args[1:], **kwargs)


class JobsListener(abc.ABC):
    def __init__(self, f=print, refresh_time=1):
        self.refresh_time = refresh_time
        self.f = f

    def receive(self, *args):
        jobs_list = args[0]
        # todo: check this
        jobs_pend = sum(["Pending" in elem in elem for elem in jobs_list])
        job_run = sum(["Running" in elem for elem in jobs_list])
        self.f(
            f"> There are {job_run} jobs running and {jobs_pend} waiting over a total of {len(jobs_list)} jobs."
        )
        # Check if some jobs have failed
        jobs_failed = [elem for elem in jobs_list if "Error" in elem]
        for job in jobs_failed:
            self.f(job)


class JobPool:
    _next_pool_id = 0
    _pool_owners = {}

    def __init__(self, scaffold):
        self._running_futures: list[concurrent.futures.Future] = []
        self._pool: typing.Optional["MPIExecutor"] = None
        self._job_queue = []
        self.id = JobPool._next_pool_id
        self._listeners = []
        self._max_wait = 60
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

    def queue(self, args=None, kwargs=None, deps=None):
        job = Job(self, args or (), kwargs or {}, deps)
        self._put(job)
        return job

    def queue_placement(self, strategy, chunk, deps=None):
        job = PlacementJob(self, strategy, chunk, deps)
        self._put(job)
        return job

    def queue_function(self, f, args=None, kwargs=None, deps=None):
        job = FunctionJob(self, f, args or [], kwargs or {}, deps)
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
            # Tell each job in our queue that they have to put themselves in the pool
            # queue; each job will store their own future and will use the futures of
            # their previously enqueued dependencies to determine when they can put
            # themselves on the pool queue.
            for job in self._job_queue:
                job._enqueue(self)
            # As long as any of the jobs aren't done yet we repeat the wait action with a timeout deined
            # by min_refresh_time, a time interval variable (t) is defined to track the time passing between the refresh
            # is important that the refresh time of listeners MUST be multiple of min_refresh_time
            while self._job_queue:
                concurrent.futures.wait(
                    self._running_futures,
                    timeout=self._max_wait,
                    return_when="FIRST_COMPLETED",
                )
                # Send the updates to the listeners that have reached refresh time
                for listener in self._listeners:
                    listener(self._job_queue)
            self._pool.shutdown()
        else:
            # todo: start a thread/coroutines/asyncio that calls the listeners periodically
            # Just run each job serially
            for job in self._job_queue:
                # Execute the static handler

                job.execute(self.owner, job._args, job._kwargs)

                # Trigger job completion manually as there is no async future object
                # like in parallel execution.
                job._completion(None)

            # Clear the queue after all jobs have been done
            self._job_queue = []
