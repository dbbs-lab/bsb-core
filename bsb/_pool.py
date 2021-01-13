"""
Job pooling module
"""

from mpi4py.MPI import COMM_WORLD


def scheduler(job):
    return job.execute()


class FakeFuture:
    def done(self):
        return False

    def running(self):
        return False


class Job:
    def __init__(self, pool, f, *args, deps=None):
        self.pool_id = pool.id
        self.f = f
        self._args = args
        self._deps = set(deps or [])
        for j in self._deps:
            j.on_completion(self._dep_completed)
        self._future = FakeFuture()

    def serialize(self):
        return (self.f, *self._args)

    def args(self):
        return self._args

    def params(self):
        return ()

    def execute(self):
        scaffold = JobPool.get_owner(self.pool_id)
        return self.f(scaffold, *self.params(), *self.args())

    def on_completion(self, cb):
        self._completion_cbs.append(cb)

    def _completion(self, _):
        for cb in self._completion_cbs:
            cb(self)

    def _dep_completed(self, dep):
        self._deps.discard(dep)
        if not self._deps:
            # Trigger the enqueueing again.
            self._enqueue(self._pool)

    def _enqueue(self, pool):
        if not self._deps:
            # Go ahead and submit ourselves to the pool, no dependencies to wait for
            self._future = pool.submit(scheduler, self)
            self._future.add_done_callback(self._completion)
            scaffold = JobPool.get_owner(self.pool_id)
            finished = lambda _: scaffold.job_finished(self)
            self._future.add_done_callback(finished)
        else:
            self._pool = pool


class ChunkedJob(Job):
    def __init__(self, pool, f, chunk, chunk_size, *args):
        self.chunk = chunk
        self.chunk_size = chunk_size
        super().__init__(pool, f, *args)

    def params(self):
        return self.chunk, self.chunk_size


class PlacementJob(ChunkedJob):
    def __init__(self, pool, type, chunk, chunk_size, deps=None, *args):
        self.type = type.name
        super().__init__(pool, type.placement.place.__func__, chunk, chunk_size, *args)

    def execute(self):
        scaffold = JobPool.get_owner(self.pool_id)
        placement = scaffold.cell_types[self.type].placement
        return self.f(placement, *self.params(), *self.args())


class JobPool:
    _next_pool_id = 0
    _pool_owners = {}

    def __init__(self, scaffold, write):
        self.parallel_write = write
        self._queue = []
        self.id = JobPool._next_pool_id
        JobPool._next_pool_id += 1
        JobPool._pool_owners[self.id] = scaffold

    @classmethod
    def get_owner(cls, id):
        return cls._pool_owners[id]

    def queue(self, f, *args):
        job = Job(self, f, *args)
        self._queue.append(job)
        return job

    def queue_chunk(self, f, chunk, *args):
        job = ChunkedJob(self, f, chunk, *args)
        self._queue.append(job)
        return job

    def queue_placement(self, type, chunk, deps=None, *args):
        job = PlacementJob(self, type, chunk, deps, *args)
        self._queue.append(job)
        return job

    def execute(self, master_event_loop):
        # This is implemented under the assumption that jobs are submitted to the pool
        # in dependency-first order; which should always be the case unless someone
        # submits jobs first and then starts adding things to the jobs' `._deps`
        # attribute. Which isn't expected to work.
        if COMM_WORLD.Get_size() == 1:
            # Just run each job serially
            for job in self._queue:
                job.execute()
            # Clear the queue after all jobs have been done
            self._queue = []
        else:
            from zwembad import MPIPoolExecutor

            # Create the MPI pool
            pool = MPIPoolExecutor()

            if pool.is_worker():
                # The workers will return out of the pool constructor when they receive
                # the shutdown signal from the master, they return here to prevent them
                # from all
                return
            # Tell each job in our queue that they have to put themselves in the pool
            # queue; each job will store their own future and will use the futures of
            # their previously enqueued dependencies to determine when they can put
            # themselves on the pool queue.
            for job in self._queue:
                job._enqueue(pool)

            # As long as any of the jobs aren't done yet we repeat the master_event_loop
            while any(not j._future.done() for j in self._queue):
                master_event_loop(self)
            pool.shutdown()


def create_job_pool(scaffold, write=True):
    return JobPool(scaffold, write=write)
