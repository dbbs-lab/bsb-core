from ..exceptions import IndicatorError, PlacementRelationError, PlacementError
from .. import config
from ..config import refs, types
from ..morphologies.selector import MorphologySelector
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
    geometry = config.dict(type=types.any_())
    morphologies = config.list(type=MorphologySelector)
    density_key = config.attr(type=str)


class _Noner:
    def __getattr__(self, attr):
        return None


class PlacementIndicator:
    def __init__(self, strat, cell_type):
        self._strat = strat
        self._cell_type = cell_type

    @property
    def cell_type(self):
        return self._cell_type

    def get_radius(self):
        return self.assert_indication("radius")

    def use_morphologies(self):
        return bool(self.indication("morphologies"))

    def indication(self, attr):
        ind_strat = self._strat.overrides.get(self._cell_type.name) or _Noner()
        ind_ct = self._cell_type.spatial
        strat = getattr(ind_strat, attr)
        ct = getattr(ind_ct, attr)
        if strat is not None:
            return strat
        return ct

    def assert_indication(self, attr):
        ind = self.indication(attr)
        if ind is None:
            raise IndicatorError(
                f"No configuration indicators found for the {attr}"
                f" of '{self._cell_type.name}' in '{self._strat.name}'"
            )
        return ind

    def guess(self, chunk=None, voxels=None):
        count = self.indication("count")
        density = self.indication("density")
        density_key = self.indication("density_key")
        planar_density = self.indication("planar_density")
        relative_to = self.indication("relative_to")
        density_ratio = self.indication("density_ratio")
        count_ratio = self.indication("count_ratio")
        if count is not None:
            estimate = self._estim_for_chunk(chunk, count)
        if density_key is not None:
            pass
        if density is not None:
            estimate = self._density_to_estim(density, chunk)
        if planar_density is not None:
            estimate = self._pdensity_to_estim(planar_density, chunk)
        if relative_to is not None:
            relation = relative_to
            if count_ratio is not None:
                strats = self._strat.scaffold.get_placement_of(relation)
                estimate = (
                    sum(PlacementIndicator(s, relation).guess() for s in strats)
                    * count_ratio
                )
                estimate = self._estim_for_chunk(chunk, estimate)
            elif density_ratio is not None:
                # Create an indicator based on this strategy for the related CT.
                # This means we'll read only the CT indications, and ignore any
                # overrides of other strats, but one can set overrides for the
                # related type in this strat.
                rel_ind = PlacementIndicator(self._strat, relation)
                rel_density = rel_ind.indication("density")
                rel_pl_density = rel_ind.indication("planar_density")
                if rel_density is not None:
                    estimate = self._density_to_estim(rel_density * density_ratio, chunk)
                elif rel_pl_density is not None:
                    estimate = self._pdensity_to_estim(
                        rel_pl_density * density_ratio, chunk
                    )
                else:
                    raise PlacementRelationError(
                        "%cell_type.name% requires relation %relation.name%"
                        + "to specify density information.",
                        self.cell_type,
                        relation,
                    )
            else:
                raise PlacementError(
                    "Relation specified but no ratio indications provided."
                )
        if density_key is not None:
            if voxels is None:
                raise Exception("Can't guess voxel density without a voxelset.")
            elif density_key in voxels.data_keys:
                estimate = self._estim_for_voxels(voxels, density_key)
            else:
                raise RuntimeError(
                    f"No voxel density data column '{density_key}' found in any of the"
                    " following partitions:\n"
                    + "\n".join(
                        f"* {p.name}: "
                        + (
                            fstr
                            if (
                                fstr := ", ".join(
                                    f"'{col}'" for col in p.voxelset.data_keys
                                )
                            )
                            else "no data"
                        )
                        for p in self._strat.partitions
                        if hasattr(p, "voxelset")
                    )
                    + "\n".join(
                        f"* {p.name} contains no voxelsets"
                        for p in self._strat.partitions
                        if not hasattr(p, "voxelset")
                    )
                )
        try:
            estimate = np.array(estimate)
        except NameError:
            # If `estimate` is undefined after all this then error out.
            raise IndicatorError(
                "No configuration indicators found for the number of"
                + f"'{self._cell_type.name}' in '{self._strat.name}'"
            )
        # 1.2 cells == 0.8 probability for 1, 0.2 probability for 2
        return (
            np.floor(estimate) + (np.random.rand(estimate.size) < estimate % 1)
        ).astype(int)

    def _density_to_estim(self, density, chunk=None):
        return sum(p.volume(chunk) * density for p in self._strat.partitions)

    def _pdensity_to_estim(self, planar_density, chunk=None):
        return sum(p.surface(chunk) * planar_density for p in self._strat.partitions)

    def _estim_for_chunk(self, chunk, count):
        if chunk is None:
            return count
        # When getting with absolute count for a chunk give back the count
        # proportional to the volume in this chunk vs total volume
        chunk_volume = sum(p.volume(chunk) for p in self._strat.partitions)
        total_volume = sum(p.volume() for p in self._strat.partitions)
        return count * chunk_volume / total_volume

    def _estim_for_voxels(self, voxels, key):
        return voxels.get_data(key).ravel() * np.product(
            voxels.get_size_matrix(copy=False), axis=1
        )
