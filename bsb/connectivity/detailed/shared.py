from itertools import chain

import numpy as np

from ... import config
from ...config import types
from ...exceptions import ConnectivityWarning
from ...reporting import warn
from ...storage._chunks import Chunk


class Intersectional:
    affinity = config.attr(type=types.fraction(), default=1)

    def get_region_of_interest(self, chunk):
        post_ps = [ct.get_placement_set() for ct in self.postsynaptic.cell_types]
        lpre, upre = self.presynaptic._get_rect_ext(tuple(chunk.dimensions))
        lpost, upost = self.postsynaptic._get_rect_ext(tuple(chunk.dimensions))
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


__all__ = ["Intersectional"]
