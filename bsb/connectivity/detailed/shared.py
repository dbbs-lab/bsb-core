from itertools import chain
from functools import reduce, cache
import numpy as np


class Intersectional:
    def get_region_of_interest(self, chunk):
        post_ps = [ct.get_placement_set() for ct in self.postsynaptic.cell_types]
        lpre, upre = self._get_rect_ext(chunk, True)
        lpost, upost = self._get_rect_ext(chunk, False)
        # Combine upper and lower bounds
        bounds = list(
            np.arange(l1 - u2 + c, u1 - l2 + c + 1)
            for l1, l2, u1, u2, c in zip(lpre, lpost, upre, upost, chunk)
        )
        # Flatten and stack the meshgrid coordinates into a list.
        clist = np.column_stack(
            tuple(a.reshape(-1) for a in np.meshgrid(*bounds, indexing="ij"))
        )
        if not hasattr(self, "_occ_chunks"):
            # Filter by chunks where cells were actually placed
            self._occ_chunks = set(
                chain.from_iterable(ps.get_all_chunks() for ps in post_ps)
            )
        return [t for c in clist if (t := tuple(c)) in self._occ_chunks]

    @cache
    def _get_rect_ext(self, chunk, pre_post_flag):
        if pre_post_flag:
            types = self.presynaptic.cell_types
        else:
            types = self.postsynaptic.cell_types
        ps_list = [ct.get_placement_set() for ct in types]
        ms_list = [ps.load_morphologies() for ps in ps_list]
        metas = list(chain.from_iterable(ms.iter_meta(unique=True) for ms in ms_list))
        # TODO: Combine morphology extension information with PS rotation information.
        # Get the chunk coordinates of the boundaries of this chunk convoluted with the
        # extension of the intersecting morphologies.
        lbounds = np.min([m["ldc"] for m in metas], axis=0) // chunk.dimensions
        ubounds = np.max([m["mdc"] for m in metas], axis=0) // chunk.dimensions
        return lbounds, ubounds

    def candidate_intersection(self, target_coll, candidate_coll):
        target_cache = [
            (ttype, tset, tset.load_boxes())
            for ttype, tset in target_coll.placement.items()
        ]
        for ctype, cset in candidate_coll.placement.items():
            box_tree = cset.load_box_tree()
            print("Target has,", len(box_tree), "boxes")
            for ttype, tset, tboxes in target_cache:
                yield (tset, cset, box_tree.query(tboxes))
