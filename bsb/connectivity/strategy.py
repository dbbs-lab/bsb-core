import abc
import typing
from itertools import chain

from .. import config
from .._util import ichain, obj_str_insert
from ..config import refs, types
from ..exceptions import ConnectivityError
from ..mixins import HasDependencies
from ..profiling import node_meter
from ..reporting import warn

if typing.TYPE_CHECKING:
    from ..cell_types import CellType
    from ..core import Scaffold
    from ..morphologies import MorphologySet
    from ..services import JobPool
    from ..storage.interfaces import PlacementSet


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
    """
    Function to load the morphologies (MorphologySet) from a PlacementSet. This override
    can allow temporary dynamic morphology generation during the connectivity phase, from
    a much smaller, or empty, MorphologySet. It is useful for example when the task would
    take too much disk space or time otherwise.
    """


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
class ConnectionStrategy(abc.ABC, HasDependencies):
    scaffold: "Scaffold"
    name: str = config.attr(key=True)
    """Name used to refer to the connectivity strategy"""
    presynaptic: Hemitype = config.attr(type=Hemitype, required=True)
    """Presynaptic (source) neuron population"""
    postsynaptic: Hemitype = config.attr(type=Hemitype, required=True)
    """Postsynaptic (target) neuron population"""
    depends_on: list["ConnectionStrategy"] = config.reflist(refs.connectivity_ref)
    """The list of strategies that must run before this one"""
    output_naming: typing.Union[str, None, dict[str, dict[str, str, None, list[str]]]] = (
        config.attr(
            type=types.or_(
                types.str(),
                types.dict(
                    type=types.dict(
                        type=types.or_(
                            types.str(), types.list(type=types.str()), types.none()
                        )
                    )
                ),
                types.list(type=types.str()),
            )
        )
    )
    """Specifies how to name the output ConnectivitySets in which the connections between cell type pairs are stored."""

    def __init_subclass__(cls, **kwargs):
        super(cls, cls).__init_subclass__(**kwargs)
        # Decorate subclasses to measure performance
        node_meter("connect")(cls)

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        # This comparison should sort connection strategies by name, via __repr__ below
        return str(self) < str(other)

    @obj_str_insert
    def __repr__(self):
        if not hasattr(self, "scaffold"):
            return f"'{self.name}'"
        pre = [ct.name for ct in self.presynaptic.cell_types]
        post = [ct.name for ct in self.postsynaptic.cell_types]
        return f"'{self.name}', connecting {pre} to {post}"

    @abc.abstractmethod
    def connect(self, presyn_collection, postsyn_collection):
        pass

    def get_deps(self):
        return set(self.depends_on)

    def _get_connect_args_from_job(self, pre_roi, post_roi):
        pre = HemitypeCollection(self.presynaptic, pre_roi)
        post = HemitypeCollection(self.postsynaptic, post_roi)
        return pre, post

    def connect_cells(self, pre_set, post_set, src_locs, dest_locs, tag=None):
        names = self.get_output_names(pre_set.cell_type, post_set.cell_type)
        between_msg = f"between {pre_set.cell_type.name} and {post_set.cell_type.name}"
        if len(names) == 0:
            raise ConnectivityError(
                f"Connections {between_msg} have been disabled by output naming."
            )
        elif len(names) == 1:
            name = names[0]
            if tag is not None and tag != name:
                raise ConnectivityError(
                    f"Tag ('{tag}') and output name ('{name}') mismatch."
                )
        else:
            names_msg = f"{between_msg} (names: {', '.join(names)})."
            if tag is None:
                raise ConnectivityError(
                    f"No tag was given to decide between multiple output names {names_msg}"
                )
            elif tag not in names:
                raise ConnectivityError(
                    f"Tag '{tag}' is not a valid output name {names_msg}"
                )
            else:
                name = tag

        cs = self.scaffold.require_connectivity_set(
            pre_set.cell_type, post_set.cell_type, name
        )
        cs.connect(pre_set, post_set, src_locs, dest_locs)

    def get_region_of_interest(self, chunk):
        pass

    def queue(self, pool: "JobPool"):
        """
        Specifies how to queue this connectivity strategy into a job pool. Can
        be overridden, the default implementation asks each partition to chunk
        itself and creates 1 placement job per chunk.
        """
        # Get the queued jobs of all the strategies we depend on.
        dep_jobs = set(
            chain.from_iterable(
                pool.get_submissions_of(strat) for strat in self.get_deps()
            )
        )
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
            job = pool.queue_connectivity(self, [chunk], roi, deps=dep_jobs)

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

    def get_output_names(self, pre=None, post=None):
        if (pre is None) != (post is None):
            raise RuntimeError("pre and post must be specified or omitted together.")
        if pre is not None and (
            pre not in self.presynaptic.cell_types
            or post not in self.postsynaptic.cell_types
        ):
            raise ValueError(
                f"'{pre.name}' and '{post.name}' are not a valid cell pair type for this connectivity strategy."
            )
        if self.output_naming is None or isinstance(self.output_naming, str):
            return self._infer_output_name(self.output_naming or self.name, pre, post)
        elif isinstance(self.output_naming, list):
            # Call `_infer_output_name` for each given `base` in the list, and chain them together
            return [
                *ichain(
                    self._infer_output_name(base, pre, post)
                    for base in self.output_naming
                )
            ]
        else:
            return self._get_output_name(pre, post)

    def _infer_output_name(self, base, pre, post):
        if len(self.presynaptic.cell_types) > 1 or len(self.postsynaptic.cell_types) > 1:
            if pre is None:
                # All output names
                return [
                    *ichain(
                        self._infer_output_name(base, pre_ct, post_ct)
                        for pre_ct in self.presynaptic.cell_types
                        for post_ct in self.postsynaptic.cell_types
                    )
                ]
            else:
                # Pair specific output name
                return [f"{base}_{pre.name}_to_{post.name}"]
        else:
            # Single output name
            return [base]

    def _get_output_name(self, pre, post):
        if pre is None:
            # All output names
            return [
                *ichain(
                    self._get_output_name(pre_ct, post_ct)
                    for pre_ct in self.presynaptic.cell_types
                    for post_ct in self.postsynaptic.cell_types
                )
            ]
        else:
            # Pair specific output name
            MISSING = type("MISSING", (), {"get": lambda *args: MISSING})()
            spec = self.output_naming.get(pre.name, MISSING).get(post.name, MISSING)
            if spec is MISSING:
                return self._infer_output_name(self.name, pre, post)
            elif spec is None:
                return []
            elif isinstance(spec, str):
                return [spec]
            else:
                return spec


__all__ = ["ConnectionStrategy", "Hemitype", "HemitypeCollection"]
