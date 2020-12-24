"""
Job pooling module
"""

from zwembad import Pool
from mpi4py.MPI import COMM_WORLD


def scheduler(job):
    return job.execute()


class Job:
    def __init__(self, pool, f, *args):
        self.pool_id = pool.id
        self.f = f
        self._args = args

    def serialize(self):
        return (self.f, *self._args)

    def args(self):
        return self._args

    def params(self):
        return ()

    def execute(self):
        scaffold = JobPool.get_owner(self.pool_id)
        return self.f(scaffold, *self.params(), *self.args())


class ChunkedJob(Job):
    def __init__(self, pool, f, chunk, chunk_size, *args):
        self.chunk = chunk
        self.chunk_size = chunk_size
        super().__init__(pool, f, *args)

    def params(self):
        return self.chunk, self.chunk_size


class PlacementJob(ChunkedJob):
    def __init__(self, pool, type, chunk, chunk_size, *args):
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
        self._queue.append(Job(self, f, *args))

    def queue_chunk(self, f, chunk, *args):
        self._queue.append(ChunkedJob(self, f, chunk, *args))

    def queue_placement(self, type, chunk, *args):
        if type.name != "dcn_cell":
            return
        self._queue.append(PlacementJob(self, type, chunk, *args))

    def selfsum(self, a, b):
        return a + b

    def execute(self):
        if COMM_WORLD.Get_size() == 1:
            # Use serial map instead of an MPI pool to run the jobs
            r = list(map(scheduler, self._queue))
        else:
            # Create the MPI job pool
            pool = Pool()
            # Run all the jobs
            r = pool.map(scheduler, self._queue)
        # Clear the queue after all jobs have been done
        self._queue = []
        return r


def create_job_pool(scaffold, write=True):
    return JobPool(scaffold, write=write)
