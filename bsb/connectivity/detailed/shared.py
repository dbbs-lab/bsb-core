from itertools import chain
from functools import reduce
import numpy as np


class Intersectional:
    def get_region_of_interest(self, chunk, chunk_size):
        post_ps = [ct.get_placement_set() for ct in self.postsynaptic.cell_types]
        lpre, upre = self._get_rect_ext(chunk, chunk_size, self.presynaptic.cell_types)
        lpost, upost = self._get_rect_ext(chunk, chunk_size, self.postsynaptic.cell_types)
        # Combine upper and lower bounds
        bounds = list(
            np.arange(l1 - u2 + c, u1 - l2 + c + 1)
            for l1, l2, u1, u2, c in zip(lpre, lpost, upre, upost, chunk)
        )
        # Flatten and stack the meshgrid coordinates into a list.
        clist = np.column_stack(
            tuple(a.reshape(-1) for a in np.meshgrid(*bounds, indexing="ij"))
        )
        # Filter by chunks where cells were actually placed
        occupied_chunks = set(chain.from_iterable(ps.get_all_chunks() for ps in post_ps))
        return [t for c in clist if (t := tuple(c)) in occupied_chunks]

    def _get_rect_ext(self, chunk, chunk_size, types):
        ps_list = [ct.get_placement_set() for ct in types]
        ms_list = [ps.load_morphologies() for ps in ps_list]
        metas = list(chain.from_iterable(ms.iter_meta(unique=True) for ms in ms_list))
        # TODO: Combine morphology extension information with PS rotation information.
        _min = reduce(lambda a, b: tuple(map(min, a, b["lsb"])), metas, (0, 0, 0))
        _max = reduce(lambda a, b: tuple(map(max, a, b["msb"])), metas, (0, 0, 0))
        # Get the chunk coordinates of the boundaries of this chunk convoluted with the
        # extension of the intersecting morphologies.
        lbounds = np.floor(np.array(_min) / chunk_size)
        ubounds = np.ceil(np.array(_max) / chunk_size)
        return lbounds, ubounds
