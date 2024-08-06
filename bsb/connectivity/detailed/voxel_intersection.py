import itertools
import random

import numpy as np
from numpy.random import default_rng

from ... import config
from ..._util import ichain
from ...config import types
from ..strategy import ConnectionStrategy
from .shared import Intersectional

_rng = default_rng()


@config.node
class VoxelIntersection(Intersectional, ConnectionStrategy):
    """
    This strategy finds overlap between voxelized morphologies.

    :param contacts: number or distribution determining the amount of synaptic contacts one cell will form on another
    :param voxel_pre: the number of voxels into which the morphology will be subdivided.
    :param voxel_post: the number of voxels into which the morphology will be subdivided.
    :param favor_cache: choose whether to cache the pre or post morphology.
    """

    contacts = config.attr(type=types.distribution(), default=1)
    voxels_pre = config.attr(type=int, default=50)
    voxels_post = config.attr(type=int, default=50)
    cache = config.attr(type=bool, default=True)
    favor_cache = config.attr(type=types.in_(["pre", "post"]), default="pre")

    def connect(self, pre, post):
        # Note on the caching terms: `targets` are the population that will be cached the
        # strongest; their voxelized tree will remain in place, while the candidates are
        # rotated and translated to overlap the target tree.
        # The choice to make something cached harder is if they have less different
        # morphologies, and good choices for candidates are the population with more
        # numerous and smaller morphologies.
        if self.favor_cache == "pre":
            targets = pre
            candidates = post
            self._n_tvoxels = self.voxels_pre
            self._n_cvoxels = self.voxels_post
            target_morpho = self.presynaptic.morpho_loader
            cand_morpho = self.postsynaptic.morpho_loader
        else:
            targets = post
            candidates = pre
            self._n_tvoxels = self.voxels_post
            self._n_cvoxels = self.voxels_pre
            target_morpho = self.postsynaptic.morpho_loader
            cand_morpho = self.presynaptic.morpho_loader
        combo_itr = self.candidate_intersection(targets, candidates)
        mset_cache = {}
        for target_set, cand_set, match_itr in combo_itr:
            if self.cache:
                if id(target_set) not in mset_cache:
                    mset_cache[id(target_set)] = target_morpho(target_set)
                if id(cand_set) not in mset_cache:
                    mset_cache[id(cand_set)] = cand_morpho(cand_set)
                target_mset = mset_cache[id(target_set)]
                cand_mset = mset_cache[id(cand_set)]
            else:
                target_mset = target_morpho(target_set)
                cand_mset = cand_morpho(cand_set)
            self._match_voxel_intersection(
                match_itr, target_set, cand_set, target_mset, cand_mset
            )

    def _match_voxel_intersection(self, matches, tset, cset, tmset, cmset):
        # Soft-caching caches at the IO level and gives you a fresh copy of the morphology
        # each time, the `cached_voxelize` function we need wouldn't have any effect!
        tm_iter = tmset.iter_morphologies(cache=self.cache, hard_cache=self.cache)
        target_itrs = zip(tset.load_positions(), tset.load_rotations().iter(), tm_iter)
        rotations = cset.load_rotations()
        positions = cset.load_positions()
        data_acc = []
        for target, candidates in enumerate(matches):
            tpos, trot, tmor = next(target_itrs)
            if not len(candidates):
                # No need to load or voxelize if there's no candidates anyway
                continue
            # Load and voxelize the target into a box tree
            if self.cache:
                tvoxels = tmor.cached_voxelize(N=self._n_tvoxels)
            else:
                tvoxels = tmor.voxelize(N=self._n_tvoxels)
            tree = tvoxels.as_boxtree(cache=self.cache)
            for cand in candidates:
                cpos = positions[cand]
                crot = rotations[cand]
                # Don't hard cache, as we mutate the instance we get.
                morpho = cmset.get(cand, cache=self.cache, hard_cache=False)
                # Transform candidate, keep target unrotated and untranslated at origin:
                # 1) Rotate self by own rotation
                # 2) Translate by position relative to target
                # 3) Anti-rotate by target rotation
                # Gives us the candidate relative to the target without having to modify,
                # reload, recalculate or revoxelize any of the target morphologies.
                # So in the case of a single target morphology we can keep that around.
                morpho.rotate(crot)
                morpho.translate(cpos - tpos)
                morpho.rotate(trot.inv())
                cvoxels = morpho.voxelize(N=self._n_cvoxels)
                boxes = cvoxels.as_boxes()
                # Filter out the candidate voxels that overlap with target voxels.
                overlap = [(i, v) for i, v in enumerate(tree.query(boxes)) if v]
                if overlap:
                    locations = self._pick_locations(
                        target, cand, tvoxels, cvoxels, overlap
                    )
                    data_acc.append(locations)

        # Preallocating and filling is faster than `np.concatenate` :shrugs:
        acc_idx = np.cumsum(
            [len(a[0]) for a in data_acc],
        )
        # The inline if guards against the case where there's no overlap
        tlocs = np.empty((acc_idx[-1] if len(acc_idx) else 0, 3), dtype=int)
        clocs = np.empty((acc_idx[-1] if len(acc_idx) else 0, 3), dtype=int)
        for (s, e), (tblock, cblock) in zip(_pairs_with_zero(acc_idx), data_acc):
            tlocs[s:e] = tblock
            clocs[s:e] = cblock

        if self.favor_cache == "pre":
            src_set, dest_set = tset, cset
            src_locs, dest_locs = tlocs, clocs
        else:
            src_set, dest_set = cset, tset
            src_locs, dest_locs = clocs, tlocs

        self.connect_cells(src_set, dest_set, src_locs, dest_locs)

    def _pick_locations(self, tid, cid, tvoxels, cvoxels, overlap):
        n = int(self.contacts.draw(1)[0])
        if n <= 0:
            return np.empty((0, 3), dtype=int), np.empty((0, 3), dtype=int)
        cpool = cvoxels.get_data([c for c, _ in overlap])
        tpool = [tvoxels.get_data(t) for _, t in overlap]
        pool = np.column_stack(
            (
                np.repeat(cpool, [len(t) for t in tpool]),
                np.array([*ichain(tpool)], dtype=object),
            )
        )
        weights = [len(c) * len(t) for c, t in pool]
        tlocs = []
        clocs = []
        for cpick, tpick in random.choices(pool, weights, k=n):
            clocs.append((cid, *random.choice(cpick)))
            tlocs.append((tid, *random.choice(tpick)))
        return tlocs, clocs


def _pairs_with_zero(iterable):
    a, b = itertools.tee(iterable)
    try:
        yield 0, next(b)
    except StopIteration:
        pass
    else:
        yield from zip(a, b)


__all__ = ["VoxelIntersection"]
