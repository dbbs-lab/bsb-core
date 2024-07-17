import typing

import numpy as np

from .. import config
from ..config import types
from ..mixins import InvertedRoI
from .strategy import ConnectionStrategy

if typing.TYPE_CHECKING:
    from ..config import Distribution


@config.node
class Convergence(ConnectionStrategy):
    """
    Connect cells based on a convergence distribution, i.e. by connecting each source cell
    to X target cells.
    """

    convergence: "Distribution" = config.attr(type=types.distribution(), required=True)

    def connect(self):
        raise NotImplementedError("Needs to be restored, please open an issue.")


class AllToAll(ConnectionStrategy):
    """
    All to all connectivity between two neural populations
    """

    def connect(self, pre, post):
        for from_ps in pre.placement:
            fl = len(from_ps)
            for to_ps in post.placement:
                len_ = len(to_ps)
                ml = fl * len_
                src_locs = np.full((ml, 3), -1)
                dest_locs = np.full((ml, 3), -1)
                src_locs[:, 0] = np.repeat(np.arange(fl), len_)
                dest_locs[:, 0] = np.tile(np.arange(len_), fl)
                self.connect_cells(from_ps, to_ps, src_locs, dest_locs)


def _connect_fixed_degree(self, pre, post, degree, is_in):
    # Generalized connect function for Fixed in- and out-degree
    rng = np.random.default_rng()
    ps_counted = pre.placement if is_in else post.placement
    ps_fixed = post.placement if is_in else pre.placement
    high = sum(len(ps) for ps in ps_counted)
    for ps in ps_fixed:
        l = len(ps)
        counted_targets = np.full((l * degree, 3), -1)
        fixed_targets = np.full((l * degree, 3), -1)
        ptr = 0
        for i in range(l):
            fixed_targets[ptr : ptr + degree, 0] = i
            counted_targets[ptr : ptr + degree, 0] = rng.choice(
                high, degree, replace=False
            )
            ptr += degree
        lowmux = 0
        for ps_o in ps_counted:
            highmux = lowmux + len(ps_o)
            demux_idx = (counted_targets[:, 0] >= lowmux) & (
                counted_targets[:, 0] < highmux
            )
            demuxed = counted_targets[demux_idx]
            demuxed[:, 0] -= lowmux
            if is_in:
                self.connect_cells(ps_o, ps, demuxed, fixed_targets[demux_idx])
            else:
                self.connect_cells(ps, ps_o, fixed_targets[demux_idx], demuxed)
            lowmux = highmux


@config.node
class FixedIndegree(InvertedRoI, ConnectionStrategy):
    """
    Connect a group of postsynaptic cell types to ``indegree`` uniformly random
    presynaptic cells from all the presynaptic cell types.
    """

    indegree: int = config.attr(type=int, required=True)

    def connect(self, pre, post):
        _connect_fixed_degree(self, pre, post, self.indegree, True)


@config.node
class FixedOutdegree(ConnectionStrategy):
    """
    Connect a group of presynaptic cell types to ``outdegree`` uniformly random
    postsynaptic cells from all the postsynaptic cell types.
    """

    outdegree: int = config.attr(type=int, required=True)

    def connect(self, pre, post):
        _connect_fixed_degree(self, pre, post, self.outdegree, False)


__all__ = ["AllToAll", "Convergence", "FixedIndegree", "FixedOutdegree"]
