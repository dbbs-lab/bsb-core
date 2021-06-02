from ..exceptions import *
from .. import config
from ..config import refs

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
            raise IndicatorError(f"No configuration indicators found for the radius of '{self._cell_type.name}'")
        return r

    def guess(self, chunk=None, chunk_size=None):
        return 20

    def _get_placement_count_old(self):
        if self.count is not None:
            return int(self._count_for_chunk(chunk, chunk_size, self.count))
        if self.density is not None:
            return self._density_to_count(self.density, chunk, chunk_size)
        if self.planar_density is not None:
            return self._pdensity_to_count(self.planar_density, chunk, chunk_size)
        if self.placement_relative_to is not None:
            relation = self.placement_relative_to.placement
            if self.placement_count_ratio is not None:
                count = relation.get_placement_count() * self.placement_count_ratio
                count = self._count_for_chunk(chunk, chunk_size, count)
            elif self.density_ratio:
                if relation.density is not None:
                    count = self._density_to_count(
                        relation.density * self.density_ratio, chunk, chunk_size
                    )
                elif relation.planar_density is not None:
                    count = self._pdensity_to_count(
                        relation.planar_density * self.density_ratio, chunk, chunk_size
                    )
                else:
                    raise PlacementRelationError(
                        "%cell_type.name% requires relation %relation.name% to specify density information.",
                        self.cell_type,
                        relation,
                    )
            if chunk is not None:
                # If we're checking the count for a specific chunk we give back a float so
                # that placement strategies  can use the decimal value to roll to add
                # stragglers and get better precision at high chunk counts and low
                # densities.
                return count
            else:
                # If we're checking total count round the number to an int; can't place
                # half cells.
                return int(count)

    def _density_to_count(self, density, chunk=None, size=None):
        return sum(p.volume(chunk, size) * density for p in self.partitions)

    def _pdensity_to_count(self, planar_density, chunk=None, size=None):
        return sum(p.surface(chunk, size) * planar_density for p in self.partitions)

    def _count_for_chunk(self, chunk, chunk_size, count):
        if chunk is None:
            return count
        # When getting with absolute count for a chunk give back the count
        # proportional to the volume in this chunk vs total volume
        chunk_volume = sum(p.volume(chunk, chunk_size) for p in self.partitions)
        total_volume = sum(p.volume() for p in self.partitions)
        return count * chunk_volume / total_volume

    def add_stragglers(self, chunk, chunk_size, chunk_count):
        """
        Adds extra cells when the number of cells can't be exactly divided over the number
        of chunks. Default implementation will take the ``chunk_count`` and use the
        decimal value as a random roll to add an extra cell.

        Example
        -------

        5 chunks have to place 6 cells; so each chunk is told to place 1.2 cells so each
        chunk will place 1 cell and have a 0.2 chance to place an extra straggler.

        This function will then return either 1 or 2 to each chunk that asks to add
        stragglers, depending on the outcome of the 0.2 chance roll.
        """
        return int(np.floor(chunk_count) + (np.random.rand() <= chunk_count % 1))
