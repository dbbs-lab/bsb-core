from ..exceptions import *
from .. import config
from ..config import refs
import numpy as np

@config.node
class PlacementIndications:
    radius = config.attr(type=float)
    density = config.attr(type=float)
    planar_density = config.attr(type=float)
    count_ratio = config.attr(type=float)
    density_ratio = config.attr(type=float)
    relative_to = config.ref(refs.cell_type_ref)
    count = config.attr(type=int)

class _Noner:
    def __getattr__(self, attr):
        return None

class PlacementIndicator:
    def __init__(self, strat, cell_type):
        self._strat = strat
        self._cell_type = cell_type
        self._ind_strat = strat.overrides.get(cell_type.name) or _Noner()
        self._ind_ct = cell_type.spatial

    def get_radius(self):
        r = self._ind_strat.radius or self._ind_ct.radius
        if r is None:
            raise IndicatorError(f"No configuration indicators found for the radius of '{self._cell_type.name}' in '{self._strat.name}'")
        return r

    def indication(self, attr):
        strat = getattr(self._ind_strat, attr)
        ct = getattr(self._ind_ct, attr)
        if strat is not None:
            return strat
        return ct

    def guess(self, chunk=None, chunk_size=None):
        count = self.indication("count")
        density = self.indication("density")
        planar_density = self.indication("planar_density")
        relative_to = self.indication("relative_to")
        density_ratio = self.indication("density_ratio")
        count_ratio = self.indication("count_ratio")
        if count is not None:
            estimate = self._estim_for_chunk(chunk, chunk_size, count)
        if density is not None:
            estimate = self._density_to_estim(density, chunk, chunk_size)
        if planar_density is not None:
            estimate = self._pdensity_to_estim(planar_density, chunk, chunk_size)
        if relative_to is not None:
            relation = relative_to.spatial
            if count_ratio is not None:
                estimate = PlacementIndicator(self, relation).guess() * count_ratio
                estimate = self._estim_for_chunk(chunk, chunk_size, estim)
            elif density_ratio is not None:
                if relation.density is not None:
                    estimate = self._density_to_estim(
                        relation.density * density_ratio, chunk, chunk_size
                    )
                elif relation.planar_density is not None:
                    estimate = self._pdensity_to_estim(
                        relation.planar_density * density_ratio, chunk, chunk_size
                    )
                else:
                    raise PlacementRelationError(
                        "%cell_type.name% requires relation %relation.name% to specify density information.",
                        self.cell_type,
                        relation,
                    )
            else:
                raise PlacementError("Relation specified but no ratio indications provided.")
        try:
            # 1.2 cells == 0.8p for 1, 0.2p for 2
            return int(np.floor(estimate) + (np.random.rand() < estimate % 1))
        except NameError:
            # If `estimate` is undefined after all this then there were no indicators.
            raise IndicatorError(f"No configuration indicators found for the number of '{self._cell_type.name}' in '{self._strat.name}'")

    def _density_to_estim(self, density, chunk=None, size=None):
        return sum(p.volume(chunk, size) * density for p in self._strat.partitions)

    def _pdensity_to_estim(self, planar_density, chunk=None, size=None):
        return sum(p.surface(chunk, size) * planar_density for p in self._strat.partitions)

    def _estim_for_chunk(self, chunk, chunk_size, count):
        if chunk is None:
            return count
        # When getting with absolute count for a chunk give back the count
        # proportional to the volume in this chunk vs total volume
        chunk_volume = sum(p.volume(chunk, chunk_size) for p in self._strat.partitions)
        total_volume = sum(p.volume() for p in self._strat.partitions)
        return count * chunk_volume / total_volume
