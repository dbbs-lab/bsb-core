import abc as _abc
import itertools
import typing
from graphlib import TopologicalSorter

from . import _util as _gutil
from .reporting import warn
from .storage._chunks import Chunk

if typing.TYPE_CHECKING:
    from .services import JobPool


def _queue_placement(self, pool: "JobPool", chunk_size):
    # Get the queued jobs of all the strategies we depend on.
    deps = set(
        itertools.chain(*(pool.get_submissions_of(strat) for strat in self.get_deps()))
    )
    # todo: perhaps pass the volume or partition boundaries as chunk size
    pool.queue_placement(self, Chunk([0, 0, 0], None), deps=deps)


def _all_chunks(iter_):
    return _gutil.unique(
        _gutil.ichain(ct.get_placement_set().get_all_chunks() for ct in iter_)
    )


def _queue_connectivity(self, pool: "JobPool"):
    """
    Get the queued jobs of all the strategies we depend on.

    param pool: pool where the jobs will be queued
    type pool: bsb.services.pool.JobPool
    """
    deps = set(_gutil.ichain(pool.get_submissions_of(strat) for strat in self.get_deps()))
    # Schedule all chunks in 1 job
    pre_chunks = _all_chunks(self.presynaptic.cell_types)
    post_chunks = _all_chunks(self.postsynaptic.cell_types)
    job = pool.queue_connectivity(self, pre_chunks, post_chunks, deps=deps)


def _raise_na(*args, **kwargs):
    raise NotImplementedError("NotParallel connection strategies have no RoI.")


class HasDependencies:
    """
    Mixin class to mark that this node may depend on other nodes.
    """

    @_abc.abstractmethod
    def get_deps(self):
        pass

    @_abc.abstractmethod
    def __lt__(self, other):
        raise NotImplementedError(f"{type(self).__name__} must implement __lt__.")

    @_abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError(f"{type(self).__name__} must implement __hash__.")

    @classmethod
    def sort_deps(cls, objects):
        """
        Orders a given dictionary of objects by the class's default mechanism and
        then apply the `after` attribute for further restrictions.
        """
        objects = set(objects)
        ordered = []
        sorter = TopologicalSorter(
            {o: set(d for d in o.get_deps() if d in objects) for o in objects}
        )
        sorter.prepare()
        while sorter.is_active():
            node_group = sorter.get_ready()
            ordered.extend(sorted(node_group))
            sorter.done(*node_group)
        return ordered


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


class InvertedRoI:
    """
    This mixin inverts the perspective of the ``get_region_of_interest`` interface and
    lets you find presynaptic regions of interest for a postsynaptic chunk.

    Usage:

    ..code-block:: python

        class MyConnStrat(InvertedRoI, ConnectionStrategy):
          def get_region_of_interest(post_chunk):
            return [pre_chunk1, pre_chunk2]
    """

    def queue(self, pool):
        # Get the queued jobs of all the strategies we depend on.
        deps = set(
            _gutil.ichain(pool.get_submissions_of(strat) for strat in self.get_deps())
        )
        post_types = self.postsynaptic.cell_types
        # Iterate over each chunk that is populated by our postsynaptic cell types.
        to_chunks = set(
            _gutil.ichain(ct.get_placement_set().get_all_chunks() for ct in post_types)
        )
        rois = {
            chunk: roi
            for chunk in to_chunks
            if (roi := self.get_region_of_interest(chunk)) is None or len(roi)
        }
        if not rois:
            warn(
                f"No overlap found between {[post.name for post in post_types]} and "
                f"{[pre.name for pre in self.presynaptic.cell_types]} "
                f"in '{self.name}'."
            )
        for chunk, roi in rois.items():
            pool.queue_connectivity(self, roi, [chunk], deps=deps)


__all__ = ["HasDependencies", "InvertedRoI", "NotParallel"]
