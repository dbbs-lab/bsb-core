import numpy as np
from ..strategy import ConnectionStrategy
from .shared import Intersectional
from ...exceptions import *
from ... import config
from ...config import types


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

    def validate(self):
        pass

    def connect(self, pre, post):
        raise NotImplementedError("under construction")
        if self.cache:
            cache = {
                set_: set_.load_morphologies()
                for set_ in itertools.chain(
                    pre.placement.values(), post.placement.values()
                )
            }
        if self.favor_cache == "pre":
            targets = self.pre
            candidates = self.post
        else:
            targets = self.post
            candidates = self.pre
        combo_itr = self.candidate_intersection(targets, candidates)
        for target_set, cand_set, match_itr in combo_itr:
            if self.cache:
                target_mset = cache[target_set]
                cand_mset = cache[cand_set]
            else:
                target_mset = target_set.load_morphologies()
                cand_mset = cand_set.load_morphologies()
            self._match_voxel_intersection(
                match_itr, target_set, cand_set, target_mset, cand_mset
            )

    def _match_voxel_intersection(self, matches, tset, cset, tmset=None, cmset=None):
        if tmset is None:
            tmset = tset.load_morphologies()
        if cmset is None:
            cmset = cset.load_morphologies()
        if self.cache:
            load = lambda m: m.cached_load
            voxelize = lambda m: m.cached_voxelize
        else:
            load = lambda m: m.load
            voxelize = lambda m: m.voxelize

        target_itrs = zip(tset.load_positions(), tset.load_rotations(), tmset)
        rotations = cset.load_rotations().cache()
        positions = cset.load_positions()
        for target, candidates in enumerate(matches):
            tpos, trot, tsm = next(target_itrs)
            if not candidates:
                # No need to load or voxelize if there's no candidates anyway
                continue
            # Load and voxelize the target into a box tree
            voxels = voxelize(load(tsm)())(N=self._n_tvoxels)
            tree = voxels.as_boxtree(cache=self.cache)
            for cand in candidates:
                cpos = positions[cand]
                crot = rotations[cand]
                morpho = load(cmset[cand])()
                if self.cache:
                    # Don't mutate the cached version
                    morpho = morpho.copy()
                # Transform relative to target:
                # 1) Rotate self by own rotation
                # 2) Translate by position relative to target
                # 3) Anti-rotate by target rotation
                # Gives us the candidate relative to the target without having to modify,
                # reload, recalculate or revoxelize any of the target morphologies.
                # So in the case of a single target morphology we can keep that around.
                morpho.rotate(crot)
                morpho.translate(cpos - tpos)
                morpho.rotate(-trot)
                voxels = morpho.voxelize(N=self._n_cvoxels)
                overlap = tree.query(voxels.as_coords(interleaved=False))
                target_voxels = [i for i, v in enumerate(overlap) if v]
                candidate_voxels = set(itertools.chain.from_iterable(overlap))
                data = voxels.get_data(candidate_voxels)
