from functools import cache
from itertools import chain

import numpy as np

from ... import config
from ...config import types
from ...exceptions import ConnectivityWarning
from ...reporting import warn
from ...storage import Chunk


class Intersectional:
    affinity = config.attr(type=types.fraction(), default=1)

    def get_region_of_interest(self, chunk):
        post_ps = [ct.get_placement_set() for ct in self.postsynaptic.cell_types]
        lpre, upre = self._get_rect_ext(tuple(chunk.dimensions), True)
        lpost, upost = self._get_rect_ext(tuple(chunk.dimensions), False)
        # Get the `np.arange`s between bounds offset by the chunk position, to be used in
        # `np.meshgrid` below.
        bounds = list(
            np.arange(l1 - u2 + c, u1 - l2 + c + 1)
            for l1, l2, u1, u2, c in zip(lpre, lpost, upre, upost, chunk)
        )
        # Flatten and stack the meshgrid coordinates into a list.
        clist = np.column_stack(
            [a.reshape(-1) for a in np.meshgrid(*bounds, indexing="ij")]
        )
        if not hasattr(self, "_occ_chunks"):
            # Filter by chunks where cells were actually placed
            self._occ_chunks = set(
                chain.from_iterable(ps.get_all_chunks() for ps in post_ps)
            )
        if not self._occ_chunks:
            warn(
                f"No {', '.join(ps.tag for ps in post_ps)} were placed, skipping {self.name}",
                ConnectivityWarning,
            )
            return []
        else:
            size = next(iter(self._occ_chunks)).dimensions
            return [t for c in clist if (t := Chunk(c, size)) in self._occ_chunks]

    @cache
    def _get_rect_ext(self, chunk_size, pre_post_flag):
        if pre_post_flag:
            types = self.presynaptic.cell_types
            loader = self.presynaptic.morpho_loader
        else:
            types = self.postsynaptic.cell_types
            loader = self.postsynaptic.morpho_loader
        ps_list = [ct.get_placement_set() for ct in types]
        ms_list = [loader(ps) for ps in ps_list]
        if not sum(map(len, ms_list)):
            # No cells placed, return smallest possible RoI.
            return [np.array([0, 0, 0]), np.array([0, 0, 0])]
        metas = list(chain.from_iterable(ms.iter_meta(unique=True) for ms in ms_list))
        # TODO: Combine morphology extension information with PS rotation information.
        # Get the chunk coordinates of the boundaries of this chunk convoluted with the
        # extension of the intersecting morphologies.
        lbounds = np.min([m["ldc"] for m in metas], axis=0) // chunk_size
        ubounds = np.max([m["mdc"] for m in metas], axis=0) // chunk_size
        return lbounds, ubounds

    def candidate_intersection(self, target_coll, candidate_coll):
        target_cache = [
            (tset.cell_type, tset, tset.load_boxes()) for tset in target_coll.placement
        ]
        for cset in candidate_coll.placement:
            box_tree = cset.load_box_tree()
            for ttype, tset, tboxes in target_cache:
                yield (tset, cset, self._affinity_filter(box_tree.query(tboxes)))

    def _affinity_filter(self, query):
        if self.affinity == 1:
            return query
        else:
            aff = self.affinity

            def sizemod(q):
                ln = len(q)
                return int(np.floor(ln * aff) + (np.random.rand() < ((ln * aff) % 1)))

            return (np.random.choice(q, sizemod(q), replace=False) for q in query)
