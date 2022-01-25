from .. import config
from ..config import refs, types
from ..helpers import SortableByAfter
from ..functions import compute_intersection_slice
from ..models import ConnectivitySet
import abc
from itertools import chain


def _targetting_req(section):
    return "labels" not in section


@config.node
class HemitypeNode:
    cell_types = config.reflist(refs.cell_type_ref, required=_targetting_req)
    compartments = config.attr(type=types.list())
    labels = config.attr(type=types.list())


class ConnectionCollection:
    def __init__(self, scaffold, cell_types, roi):
        self.scaffold = scaffold
        self.cell_types = cell_types
        self.roi = roi

    @property
    def placement(self):
        return {ct: ct.get_placement_set(self.roi) for ct in self.cell_types}

    def __getattr__(self, attr):
        return self.placement[attr]

    def __getitem__(self, item):
        return self.placement[item]


@config.dynamic
class ConnectionStrategy(abc.ABC, SortableByAfter):
    name = config.attr(key=True)
    presynaptic = config.attr(type=HemitypeNode, required=True)
    postsynaptic = config.attr(type=HemitypeNode, required=True)
    after = config.reflist(refs.placement_ref)

    @classmethod
    def get_ordered(cls, objects):
        # No need to sort connectivity strategies, just obey dependencies.
        return objects

    def get_after(self):
        return [] if not self.has_after() else self.after

    def has_after(self):
        return hasattr(self, "after")

    def create_after(self):
        self.after = []

    @abc.abstractmethod
    def connect(self, presyn_collection, postsyn_collection):
        pass

    def _get_connect_args_from_job(self, chunk, chunk_size, roi):
        pre = ConnectionCollection(self.scaffold, self.presynaptic.cell_types, [chunk])
        post = ConnectionCollection(self.scaffold, self.postsynaptic.cell_types, roi)
        return pre, post

    def connect_cells(self, pre_type, post_type, src_locs, dest_locs, tag=None):
        pass

    @abc.abstractmethod
    def get_region_of_interest(self, chunk, chunk_size):
        pass

    def queue(self, pool, chunk_size):
        """
        Specifies how to queue this connectivity strategy into a job pool. Can
        be overridden, the default implementation asks each partition to chunk
        itself and creates 1 placement job per chunk.
        """
        # Reset jobs that we own
        self._queued_jobs = []
        # Get the queued jobs of all the strategies we depend on.
        deps = set(chain.from_iterable(strat._queued_jobs for strat in self.get_after()))
        pre_types = self.presynaptic.cell_types
        # Iterate over each chunk that is populated by our presynaptic cell types.
        from_chunks = set(
            chain.from_iterable(
                ct.get_placement_set().get_all_chunks() for ct in pre_types
            )
        )
        for chunk in from_chunks:
            print("Queueing chunk", chunk)
            # Find each presynaptic chunk's postsynaptic region of interest
            roi = self.get_region_of_interest(chunk, chunk_size)
            job = pool.queue_connectivity(self, chunk, chunk_size, roi, deps=deps)
            self._queued_jobs.append(job)

    def get_cell_types(self):
        return set(self.presynaptic.cell_types) | set(self.postsynaptic.cell_types)
