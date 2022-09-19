from .. import config
from ..config import refs, types
from .._util import SortableByAfter, obj_str_insert
from ..reporting import report
import abc
from itertools import chain


@config.node
class Hemitype:
    cell_types = config.reflist(refs.cell_type_ref, required=True)
    labels = config.attr(type=types.list())
    morphology_labels = config.attr(type=types.list())


class HemitypeCollection:
    def __init__(self, hemitype, roi):
        self.hemitype = hemitype
        self.roi = roi

    def __iter__(self):
        return iter(self.hemitype.cell_types)

    @property
    def placement(self):
        return {
            ct: ct.get_placement_set(
                self.roi,
                labels=self.hemitype.labels,
                morphology_labels=self.hemitype.morphology_labels,
            )
            for ct in self.hemitype.cell_types
        }

    def __getattr__(self, attr):
        return self.placement[attr]

    def __getitem__(self, item):
        return self.placement[item]


@config.dynamic(attr_name="strategy", required=True)
class ConnectionStrategy(abc.ABC, SortableByAfter):
    name = config.attr(key=True)
    presynaptic = config.attr(type=Hemitype, required=True)
    postsynaptic = config.attr(type=Hemitype, required=True)
    after = config.reflist(refs.connectivity_ref)

    def __boot__(self):
        self._queued_jobs = []

    @obj_str_insert
    def __repr__(self):
        pre = [ct.name for ct in self.presynaptic.cell_types]
        post = [ct.name for ct in self.postsynaptic.cell_types]
        return f"'{self.name}', connecting {pre} to {post}"

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

    def _get_connect_args_from_job(self, chunk, roi):
        pre = HemitypeCollection(self.presynaptic, [chunk])
        post = HemitypeCollection(self.postsynaptic, roi)
        return pre, post

    def connect_cells(self, pre_set, post_set, src_locs, dest_locs, tag=None):
        cs = self.scaffold.require_connectivity_set(
            pre_set.cell_type, post_set.cell_type, tag
        )
        cs.connect(pre_set, post_set, src_locs, dest_locs)

    @abc.abstractmethod
    def get_region_of_interest(self, chunk):
        pass

    def queue(self, pool):
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
        rois = {
            chunk: roi
            for chunk in from_chunks
            if (roi := self.get_region_of_interest(chunk))
        }
        for chunk, roi in rois.items():
            job = pool.queue_connectivity(self, chunk, roi, deps=deps)
            self._queued_jobs.append(job)

        report(f"Queued {len(self._queued_jobs)} jobs for {self.name}", level=2)

    def get_cell_types(self):
        return set(self.presynaptic.cell_types) | set(self.postsynaptic.cell_types)
