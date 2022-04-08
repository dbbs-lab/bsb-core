import numpy as np
from numpy.random import default_rng
import itertools
import random
from ..strategy import ConnectionStrategy
from .shared import Intersectional
from ...exceptions import *
from ... import config
from ...config import types

_rng = default_rng()


@config.node
class VoxelIntersection(Intersectional, ConnectionStrategy):
    """
    This strategy voxelizes morphologies into collections of cubes, thereby reducing
    the spatial specificity of the provided traced morphologies by grouping multiple
    compartments into larger cubic voxels. Intersections are found not between the
    seperate compartments but between the voxels and random compartments of matching
    voxels are connected to eachother. This means that the connections that are made
    are less specific to the exact morphology and can be very useful when only 1 or a
    few morphologies are available to represent each cell type.
    """

    affinity = config.attr(type=types.fraction(), default=1)
    contacts = config.attr(type=types.distribution(), default=1)
    voxels_pre = config.attr(type=int, default=50)
    voxels_post = config.attr(type=int, default=50)
    cache = config.attr(type=bool, default=True)
    favor_cache = config.attr(type=types.in_(["pre", "post"]), default="pre")

    def connect(self, pre, post):
        if self.cache:
            cache = {
                set_.tag: set_.load_morphologies()
                for set_ in itertools.chain(
                    pre.placement.values(), post.placement.values()
                )
            }
        if self.favor_cache == "pre":
            targets = pre
            candidates = post
            self._n_tvoxels = self.voxels_pre
            self._n_cvoxels = self.voxels_post
        else:
            targets = post
            candidates = pre
            self._n_tvoxels = self.voxels_post
            self._n_cvoxels = self.voxels_pre
        combo_itr = self.candidate_intersection(targets, candidates)
        for target_set, cand_set, match_itr in combo_itr:
            if self.cache:
                target_mset = cache[target_set.tag]
                cand_mset = cache[cand_set.tag]
            else:
                target_mset = target_set.load_morphologies()
                cand_mset = cand_set.load_morphologies()
            self._match_voxel_intersection(
                match_itr, target_set, cand_set, target_mset, cand_mset
            )

    def _match_voxel_intersection(self, matches, tset, cset, tmset, cmset):
        # Soft caching caches at the IO level and gives you a fresh copy of the morphology
        # each time, the `cached_voxelize` function we need wouldn't have any effect!
        tm_iter = tmset.iter_morphologies(cache=self.cache, hard_cache=self.cache)
        target_itrs = zip(tset.load_positions(), tset.load_rotations().iter(), tm_iter)
        rotations = cset.load_rotations()
        positions = cset.load_positions()
        data_acc = []
        for target, candidates in enumerate(matches):
            tpos, trot, tmor = next(target_itrs)
            if not candidates:
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
        acc_idx = np.cumsum([len(a[0]) for a in data_acc])
        tlocs = np.empty((acc_idx[-1], 3))
        clocs = np.empty((acc_idx[-1], 3))
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
        # TODO: There's probably some probabilistic bias here in favor of voxels with less
        # locs inside of them to be picked more often.
        n = int(self.contacts.draw(1))
        tlocs = []
        clocs = []
        for i in _rng.integers(len(overlap), size=n):
            cv, tvs = overlap[i]
            cpool = cvoxels.get_data(cv)
            tpool = np.concatenate([tvoxels.get_data(tv) for tv in tvs])
            tlocs.append((tid, *random.choice(tpool)))
            clocs.append((cid, *random.choice(cpool)))
        return tlocs, clocs


def _pairs_with_zero(iterable):
    a, b = itertools.tee(iterable)
    yield 0, next(b)
    yield from zip(a, b)
