import numpy as np

from bsb import PlacementStrategy, VoxelSet, config, pool_cache, types


@config.node
class DistributionPlacement(PlacementStrategy):
    distribution = config.attr(type=types.distribution(), required=True)
    axis: int = config.attr(type=types.int(min=0, max=2), required=False, default=2)
    direction: str = config.attr(
        type=types.in_(["positive", "negative"]), required=False, default="positive"
    )

    @pool_cache
    def draw_all_positions(self, indicator):
        all_positions = np.empty((0, 3))
        for p in indicator.get_partitions():
            num_to_place = indicator.guess(voxels=p.to_voxels())
            rand_nums = self.distribution.draw(num_to_place)
            distrib_interval = self.distribution.definition_interval(1e-9)
            rand_nums[rand_nums < distrib_interval[0]] = distrib_interval[0]
            rand_nums[rand_nums > distrib_interval[1]] = distrib_interval[1]

            normed_rand = (rand_nums - distrib_interval[0]) / np.diff(distrib_interval)
            positions = np.random.rand(num_to_place, 2)
            positions = np.hstack([positions, normed_rand[..., np.newaxis]])
            bounds = [p._data.ldc, p._data.mdc]
            positions *= np.diff(bounds, axis=0)
            if self.direction == "positive":
                positions += bounds[0]
            else:
                positions = bounds[1] - positions
            all_positions = np.concatenate([all_positions, positions])
        return all_positions

    def place(self, chunk, indicators):
        voxelset = VoxelSet([chunk], chunk.dimensions)

        for name_indic, indicator in indicators.items():
            positions = self.draw_all_positions(indicator)
            inside_chunk = voxelset.inside(positions)
            self.place_cells(indicator, positions[inside_chunk], chunk)
