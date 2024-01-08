import itertools
import os
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


@config.node
class FixedIndegree(InvertedRoI, ConnectionStrategy):
    """
    Connect a group of postsynaptic cell types to ``indegree`` uniformly random
    presynaptic cells from all the presynaptic cell types.
    """

    indegree: int = config.attr(type=int, required=True)

    def connect(self, pre, post):
        in_ = self.indegree
        rng = np.random.default_rng()
        high = sum(len(ps) for ps in pre.placement)
        for ps in post.placement:
            l = len(ps)
            pre_targets = np.full((l * in_, 3), -1)
            post_targets = np.full((l * in_, 3), -1)
            ptr = 0
            for i in range(l):
                post_targets[ptr : ptr + in_, 0] = i
                pre_targets[ptr : ptr + in_, 0] = rng.choice(high, in_, replace=False)
                ptr += in_
            lowmux = 0
            for pre_ps in pre.placement:
                highmux = lowmux + len(pre_ps)
                demux_idx = (pre_targets[:, 0] >= lowmux) & (pre_targets[:, 0] < highmux)
                demuxed = pre_targets[demux_idx]
                demuxed[:, 0] -= lowmux
                self.connect_cells(pre_ps, ps, demuxed, post_targets[demux_idx])
                lowmux = highmux
