"""
Job pooling module
"""

from mpi4py.MPI import COMM_WORLD
import time
import concurrent.futures


def dispatcher(pool_id, job_args):
    job_type, f, args, kwargs = job_args
    # Get the static job execution handler from this module
    handler = globals()[job_type].execute
    owner = JobPool.get_owner(pool_id)
    # Execute it.
    handler(owner, f, args, kwargs)


class FakeFuture:
    def done(self):
        return False

    def running(self):
        return False


class Job:
    """
    Dispatches the execution of a function through a JobPool
    """

    def __init__(self, pool, f, args, kwargs, deps=None):
        self.pool_id = pool.id
        self.f = f
        self._args = args
        self._kwargs = kwargs
        self._deps = set(deps or [])
        self._completion_cbs = []
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = FakeFuture()

    def serialize(self):
        name = self.__class__.__name__
        # First arg is to find the static `execute` method so that we don't have to
        # serialize any of the job objects themselves but can still use different handlers
        # for different job types.
        return (name, self.f, self._args, self._kwargs)

    @staticmethod
    def execute(job_owner, f, args, kwargs):
        """
        Default job handler, invokes ``f`` passing it the scaffold object that owns the
        job + the args and kwargs given at job creation.
        """
        return f(job_owner, *args, **kwargs)

    def on_completion(self, cb):
        self._completion_cbs.append(cb)

    def _completion(self, _):
        for cb in self._completion_cbs:
            cb(self)

    def _dep_completed(self, dep):
        # Earlier we registered this callback on the completion of our dependencies.
        # When a dep completes we end up here and we discard it as a dependency as it has
        # finished. When all our dependencies have been discarded we can queue ourselves.
        self._deps.discard(dep)
        if not self._deps:
            self._enqueue(self._pool)

    def _enqueue(self, pool):
        if not self._deps:
            # Go ahead and submit ourselves to the pool, no dependencies to wait for
            # The dispatcher is run on the remote worker and unpacks the data required
            # to execute the job contents.
            self._future = pool.submit(dispatcher, self.pool_id, self.serialize())
            # Invoke our completion callbacks when the future completes.
            self._future.add_done_callback(self._completion)
        else:
            # We have unfinished dependencies and should wait until we can enqueue
            # ourselves when our dependencies haved all notified us of their completion.
            self._pool = pool


class ChunkedJob(Job):
    def __init__(self, pool, f, chunk, chunk_size, deps=None):
        args = (chunk, chunk_size)
        super().__init__(pool, f, args, {}, deps=deps)


class PlacementJob(ChunkedJob):
    """
    Dispatches the execution of a chunk of a placement strategy through a JobPool.
    """

    def __init__(self, pool, type, chunk, chunk_size, deps=None):
        args = (type.name, chunk, chunk_size)
        super(ChunkedJob, self).__init__(
            pool, type.placement.place.__func__, args, {}, deps=deps
        )

    @staticmethod
    def execute(job_owner, f, args, kwargs):
        placement = job_owner.cell_types[args[0]].placement
        return f(placement, *args[1:], **kwargs)


class JobPool:
    _next_pool_id = 0
    _pool_owners = {}

    def __init__(self, scaffold, listeners=None):
        self._queue = []
        self.id = JobPool._next_pool_id
        self._listeners = listeners or []
        JobPool._next_pool_id += 1
        JobPool._pool_owners[self.id] = scaffold

    @classmethod
    def get_owner(cls, id):
        return cls._pool_owners[id]

    @property
    def owner(self):
        return self.get_owner(self.id)

    def _put(self, job):
        """
        Puts a job onto our internal queue. Putting items in the queue does not mean they
        will be executed. For this ``self.execute()`` must be called.
        """
        # Use an observer pattern to allow the creator of the pool to listen for job
        # completion. This complements the event loop in parallel execution and is
        # executed synchronously in serial execution.
        for listener in self._listeners:
            job.on_completion(listener)
        self._queue.append(job)

    def queue(self, f, args=None, kwargs=None, deps=None):
        job = Job(self, f, args or (), kwargs or {}, deps)
        self._put(job)
        return job

    def queue_chunk(self, f, chunk, chunk_size, deps=None):
        job = ChunkedJob(self, f, chunk, chunk_size, deps)
        self._put(job)
        return job

    def queue_placement(self, type, chunk, chunk_size, deps=None):
        job = PlacementJob(self, type, chunk, chunk_size, deps)
        self._put(job)
        return job

    def execute(self, master_event_loop=None):
        """
        Execute the jobs in the queue

        In serial execution this runs all of the jobs in the queue in First In First Out
        order. In parallel execution this enqueues all jobs into the MPIPool unless they
        have dependencies that need to complete first.

        :param master_event_loop: A function that is continuously calls while waiting for
            the jobs to finish in parallel execution
        :type master_event_loop: function
        """
        # This is implemented under the assumption that jobs are submitted to the pool
        # in dependency-first order; which should always be the case unless someone
        # submits jobs first and then starts adding things to the jobs' `._deps`
        # attribute. Which isn't expected to work.
        if COMM_WORLD.Get_size() == 1:
            # Just run each job serially
            for job in self._queue:
                # Execute the static handler
                job.execute(self.owner, job.f, job._args, job._kwargs)
                # Trigger job completion manually as there is no async future object
                # like in parallel execution.
                job._completion(None)
            # Clear the queue after all jobs have been done
            self._queue = []
        else:
            from zwembad import MPIPoolExecutor

            # Create the MPI pool
            pool = MPIPoolExecutor()

            if pool.is_worker():
                # The workers will return out of the pool constructor when they receive
                # the shutdown signal from the master, they return here skipping the
                # master logic.
                return
            # Tell each job in our queue that they have to put themselves in the pool
            # queue; each job will store their own future and will use the futures of
            # their previously enqueued dependencies to determine when they can put
            # themselves on the pool queue.
            for job in self._queue:
                job._enqueue(pool)

            q = self._queue.copy()
            # As long as any of the jobs aren't done yet we repeat the master_event_loop
            while (
                open_jobs := [j._future for j in self._queue if not j._future.done()]
            ) :
                if master_event_loop:
                    # If there is an event loop, run it and hand it a copy of the jobqueue
                    master_event_loop(q)
                else:
                    # If there is no event loop just let the master idle until execution
                    # has completed.
                    concurrent.futures.wait(open_jobs)
            pool.shutdown()


def create_job_pool(scaffold):
    return JobPool(scaffold)
