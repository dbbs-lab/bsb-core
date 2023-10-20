import typing

from .. import config
from ..config import refs, types
from ..profiling import node_meter
from ..reporting import report, warn
from .._util import SortableByAfter, obj_str_insert, ichain
import abc
from itertools import chain

if typing.TYPE_CHECKING:
    from ..core import Scaffold
    from ..cell_types import CellType
    from ..storage.interfaces import PlacementSet
    from ..morphologies import MorphologySet
    from ..connectivity import ConnectionStrategy


@config.node
class Hemitype:
    """
    Class used to represent one (pre- or postsynaptic) side of a connection rule.
    """

    scaffold: "Scaffold"

    cell_types: list["CellType"] = config.reflist(refs.cell_type_ref, required=True)
    """List of cell types to use in connection."""
    labels: list[str] = config.attr(type=types.list())
    """List of labels to filter the placement set by."""
    morphology_labels: list[str] = config.attr(type=types.list())
    """List of labels to filter the morphologies by."""
    morpho_loader: typing.Callable[["PlacementSet"], "MorphologySet"] = config.attr(
        type=types.function_(),
        required=False,
        call_default=False,
        default=(lambda ps: ps.load_morphologies()),
    )
    """Function to load the morphologies (MorphologySet) from a PlacementSet"""


class HemitypeCollection:
    def __init__(self, hemitype, roi):
        self.hemitype = hemitype
        self.roi = roi

    def __iter__(self):
        return iter(self.hemitype.cell_types)

    @property
    def placement(self):
        return [
            ct.get_placement_set(
                chunks=self.roi,
                labels=self.hemitype.labels,
                morphology_labels=self.hemitype.morphology_labels,
            )
            for ct in self.hemitype.cell_types
        ]


@config.dynamic(attr_name="strategy", required=True)
class ConnectionStrategy(abc.ABC, SortableByAfter):
    scaffold: "Scaffold"
    name: str = config.attr(key=True)
    """Name used to refer to the connectivity strategy"""
    presynaptic: Hemitype = config.attr(type=Hemitype, required=True)
    """Presynaptic (source) neuron population"""
    postsynaptic: Hemitype = config.attr(type=Hemitype, required=True)
    """Postsynaptic (target) neuron population"""
    after: list["ConnectionStrategy"] = config.reflist(refs.connectivity_ref)
    """
        This strategy should be executed only after all the connections in this list have
        been executed.
    """

    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("connect")(cls)

    def __boot__(self):
        self._queued_jobs = []

    @obj_str_insert
    def __repr__(self):
        if not hasattr(self, "scaffold"):
            return f"'{self.name}'"
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

    def _get_connect_args_from_job(self, pre_roi, post_roi):
        pre = HemitypeCollection(self.presynaptic, pre_roi)
        post = HemitypeCollection(self.postsynaptic, post_roi)
        return pre, post

    def connect_cells(self, pre_set, post_set, src_locs, dest_locs, tag=None):
        if len(self.presynaptic.cell_types) > 1 or len(self.postsynaptic.cell_types) > 1:
            name = f"{self.name}_{pre_set.cell_type.name}_to_{post_set.cell_type.name}"
        else:
            name = self.name
        cs = self.scaffold.require_connectivity_set(
            pre_set.cell_type, post_set.cell_type, tag if tag is not None else name
        )
        cs.connect(pre_set, post_set, src_locs, dest_locs)

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
            if (roi := self.get_region_of_interest(chunk)) is None or len(roi)
        }
        if not rois:
            warn(
                f"No overlap found between {[pre.name for pre in pre_types]} and "
                f"{[post.name for post in self.postsynaptic.cell_types]} "
                f"in '{self.name}'."
            )
        for chunk, roi in rois.items():
            job = pool.queue_connectivity(self, [chunk], roi, deps=deps)
            self._queued_jobs.append(job)
        report(f"Queued {len(self._queued_jobs)} jobs for {self.name}", level=2)

    def get_cell_types(self):
        return set(self.presynaptic.cell_types) | set(self.postsynaptic.cell_types)

    def get_all_pre_chunks(self):
        all_ps = (ct.get_placement_set() for ct in self.presynaptic.cell_types)
        chunks = set(ichain(ps.get_all_chunks() for ps in all_ps))
        return list(chunks)

    def get_all_post_chunks(self):
        all_ps = (ct.get_placement_set() for ct in self.postsynaptic.cell_types)
        chunks = set(ichain(ps.get_all_chunks() for ps in all_ps))
        return list(chunks)
