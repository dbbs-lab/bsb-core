from .reporting import report
from .storage import Chunk
from . import _util as _gutil

import itertools


def _queue_placement(self, pool, chunk_size):
    # Reset jobs that we own
    self._queued_jobs = []
    # Get the queued jobs of all the strategies we depend on.
    deps = set(itertools.chain(*(strat._queued_jobs for strat in self.get_after())))
    # todo: perhaps pass the volume or partition boundaries as chunk size
    job = pool.queue_placement(self, Chunk([0, 0, 0], None), deps=deps)
    self._queued_jobs.append(job)
    report(f"Queued serial job for {self.name}", level=2)


def _all_chunks(iter_):
    return _gutil.unique(
        itertools.chain.from_iterable(
            ct.get_placement_set().get_all_chunks() for ct in iter_
        )
    )


def _queue_connectivity(self, pool):
    # Reset jobs that we own
    self._queued_jobs = []
    # Get the queued jobs of all the strategies we depend on.
    deps = set(
        itertools.chain.from_iterable(strat._queued_jobs for strat in self.get_after())
    )
    # Schedule all chunks in 1 job
    pre_chunks = _all_chunks(self.presynaptic.cell_types)
    post_chunks = _all_chunks(self.postsynaptic.cell_types)
    job = pool.queue_connectivity(self, pre_chunks, post_chunks, deps=deps)
    self._queued_jobs.append(job)
    report(f"Queued serial job for {self.name}", level=2)


def _raise_na(*args, **kwargs):
    raise NotImplementedError("NotParallel connection strategies have no RoI.")


class NotParallel:
    def __init_subclass__(cls, **kwargs):
        from .connectivity import ConnectionStrategy
        from .placement import PlacementStrategy

        super().__init_subclass__(**kwargs)
        if PlacementStrategy in cls.__mro__:
            cls.queue = _queue_placement
        elif ConnectionStrategy in cls.__mro__:
            cls.queue = _queue_connectivity
            if "get_region_of_interest" not in cls.__dict__:
                cls.get_region_of_interest = _raise_na
        else:
            raise Exception(
                "NotParallel can only be applied to placement or "
                "connectivity strategies"
            )
