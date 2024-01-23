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

import numpy as np

from . import MPI
from ._util import ErrorModule, MockModule


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


class Job:
    """
    Dispatches the execution of a function through a JobPool
    """

    def __init__(self, pool, args, kwargs, deps=None, submitter=None):
        self.pool_id = pool.id
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._completion_cbs = []
        self._status = "Pending"
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = ""  # FakeFuture()

    def serialize(self):
        name = self.__class__.__name__
        # First arg is to find the static `execute` method so that we don't have to
        # serialize any of the job objects themselves but can still use different handlers
        # for different job types.
        return (name, self._args, self._kwargs)

    @abc.abstractmethod
    def execute(job_owner, args, kwargs):
        """
        Default job handler, invokes ``f`` passing it the scaffold object that owns the
        job + the args and kwargs given at job creation.
        """
        pass

    def on_completion(self, cb):
        self._completion_cbs.append(cb)

    def _completion(self, future):
        # todo: First update ourselves, based on future:
        #       * retrieve error/result
        #       then, publish ourselves to all our listeners:
        self.result = self._retrieve_result(future)
        for cb in self._completion_cbs:
            cb(self, err=None, result=None)

    def _retrieve_result(self, future):
        try:
            result = future.result()
            self._status = "Complete"
            return result
        except Exception as e:
            if hasattr(self, "_cname") and hasattr(self, "_name"):
                msg = f"Error for {self._name} in {self._cname} : {e} "
            else:
                msg = f"Job {self} raise Error : {e}"
            self._status = msg

    def _dep_completed(self, dep, err=None, result=None):
        # todo: use err/result
        #       * todo maybe: write result to file, then read file for data in dependency submit
        # Earlier we registered this callback on the completion of our dependencies.
        # When a dep completes we end up here and we discard it as a dependency as it has
        # finished. If the dep returns an error remove the job from the pool, since the dependency have failed.
        self._deps.discard(dep)
        if "Error" in dep._status:
            self._deps.append(True)  # From now on this job will never be submitted
            if self._cname and self._name:
                msg = f"Error for {self._name} in {self._cname} : Job killed for dependency failure."
            else:
                msg = f"Error : Job {self} killed for dependency failure."
            self._status = msg
        else:
            # When all our dependencies have been discarded we can queue ourselves. Unless the
            # pool is serial, then the pool itself just runs all jobs in order.
            if not self._deps and MPI.get_size() > 1:
                # self._pool is set when the pool first tried to enqueue us, but we were still
                # waiting for deps, in the `_enqueue` method below.
                self._enqueue(self._pool)

    def _enqueue(self, pool):
        if not self._deps:
            # Notify anyone waiting on the spaceholder `FakeFuture` that we're
            # now actually queueing ourselves
            # self._future.set_result("ENQUEUED")
            # Go ahead and submit ourselves to the pool, no dependencies to wait for
            # The dispatcher is run on the remote worker and unpacks the data required
            # to execute the job contents.
            self._future = pool.submit(dispatcher, self.pool_id, self.serialize())
            # Invoke our completion callbacks when the future completes.
            self._future.add_done_callback(self._completion)
        else:
            # We have unfinished dependencies and should wait until we can enqueue
            # ourselves when our dependencies haved all notified us of their completion.
            # Store the reference to the pool though, so later in `_dep_completed` we can
            # call `_enqueue` again ourselves!
            self._pool = pool


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


class PipelineJob(Job):
    def __init__(self, pool, f, args, kwargs, deps=None):
        self._f = f
        new_args = [f]
        new_args.extend(args)
        super().__init__(pool, new_args, kwargs, deps=deps)

    @staticmethod
    def execute(job_owner, args, kwargs):
        f = args[0]
        return f(job_owner, *args[1:], **kwargs)


class JobsListener:
    def __init__(self, f=print, refresh_time=1):
        self.refresh_time = refresh_time
        self.f = f

    def receive(self, jobs_list):
        jobs_pend = sum(["Pending" in elem for elem in jobs_list])
        self.f(f"> There are still {jobs_pend} jobs running over {len(jobs_list)} jobs.")
        # Check if some jobs have failed
        jobs_failed = [elem for elem in jobs_list if "Error" in elem]
        for job in jobs_failed:
            self.f(job)


class JobPool:
    _next_pool_id = 0
    _pool_owners = {}

    def __init__(self, scaffold, listeners=None):
        self._queue = []
        self._job_status = []
        self.id = JobPool._next_pool_id
        self._listeners = listeners or []
        JobPool._next_pool_id += 1
        JobPool._pool_owners[self.id] = scaffold

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
        self._queue.append(job)

    def queue(self, args=None, kwargs=None, deps=None):
        job = Job(self, args or (), kwargs or {}, deps)
        self._put(job)
        return job

    def queue_placement(self, strategy, chunk, deps=None):
        job = PlacementJob(self, strategy, chunk, deps)
        self._put(job)
        return job

    def queue_pipeline(self, f, args=None, kwargs=None, deps=None):
        job = PipelineJob(self, f, args or [], kwargs or {}, deps)
        self._put(job)
        return job

    def queue_connectivity(self, strategy, pre_roi, post_roi, deps=None):
        job = ConnectivityJob(self, strategy, pre_roi, post_roi, deps)
        self._put(job)
        return job

    def execute(self, master_event_loop=None):
        """
        Execute the jobs in the queue

        In serial execution this runs all of the jobs in the queue in First In First Out
        order. In parallel execution this enqueues all jobs into the MPIPool unless they
        have dependencies that need to complete first.

        :param master_event_loop: A function that is continuously called while waiting for
          the jobs to finish in parallel execution
        :type master_event_loop: Callable
        """
        if self.parallel:
            # Create the MPI pool
            pool = _MPIPool.MPIExecutor()

            if pool.is_worker():
                # The workers will return out of the pool constructor when they receive
                # the shutdown signal from the master, they return here skipping the
                # master logic.
                return
            # Tell each job in our queue that they have to put themselves in the pool
            # queue; each job will store their own future and will use the futures of
            # their previously enqueued dependencies to determine when they can put
            # themselves on the pool queue.
            jobs_status = []
            for ind, job in enumerate(self._queue):
                job._enqueue(pool)
                jobs_status.append(job._status)
            # todo: let listeners decide `max_refresh`, and `futures.wait(timeout=max_refresh)`
            # Setting a default refresh time of 60 sec, then select the minimum between the default and
            # the refresh time of all the listeners
            ref_times = [60]
            for listener in self._listeners:
                ref_times.append(listener.refresh_time)
            min_refresh_time = min(ref_times)
            max_refresh_time = max(ref_times)
            # q = self._queue.copy()
            # As long as any of the jobs aren't done yet we repeat the wait action with a timeout deined
            # by min_refresh_time, a time interval variable (t) is defined to track the time passing between the refresh
            # is important that the refresh time of listeners MUST be multiple of min_refresh_time
            t = min_refresh_time
            while open_jobs := [j._future for j in self._queue if not j._future.done()]:
                concurrent.futures.wait(
                    open_jobs, timeout=min_refresh_time, return_when="FIRST_COMPLETED"
                )
                # Update the status for all the jobs
                for ind, job in enumerate(self._queue):
                    jobs_status[ind] = job._status

                # Send the updates to the listeners that have reached refresh time
                for listener in self._listeners:
                    if not t % listener.refresh_time:
                        listener.receive(jobs_status)
                # Update time variable
                if t == max_refresh_time:
                    t = min_refresh_time
                else:
                    t += min_refresh_time
            pool.shutdown()
        else:
            # Just run each job serially
            for job in self._queue:
                # Execute the static handler

                job.execute(self.owner, job._args, job._kwargs)

                # Trigger job completion manually as there is no async future object
                # like in parallel execution.
                job._completion(None)

            # Clear the queue after all jobs have been done
            self._queue = []


def create_job_pool(scaffold):
    return JobPool(scaffold)
