from itertools import chain
from functools import reduce


class Intersectional:
    def get_region_of_interest(self, chunk, chunk_size):
        ps_list = [ct.get_placement_set([chunk]) for ct in self.presynaptic.cell_types]
        ms_list = [ps.load_morphologies() for ps in ps_list]
        metas = list(chain.from_iterable(ms.iter_meta(unique=True) for ms in ms_list))
        # TODO: Combine morphology extension information with PS rotation information.
        _min = reduce(lambda a, b: tuple(map(min, a, b["lsb"])), metas, (0, 0, 0))
        _max = reduce(lambda a, b: tuple(map(max, a, b["msb"])), metas, (0, 0, 0))
        print(
            f"The morphologies extend maximally {_min} towards more negative coordinates"
        )
        print(
            f"The morphologies extend maximally {_max} towards more positive coordinates"
        )
