import numpy as np

from bsb import PlacementStrategy, config, types


@config.node
class DistributionPlacement(PlacementStrategy):
    distribution = config.attr(type=types.distribution(), required=True)
    axis: int = config.attr(type=types.int(min=0, max=2), required=False, default=2)
    direction: str = config.attr(
        type=types.in_(["positive", "negative"]), required=False, default="positive"
    )

    def draw_interval(self, n, lower, upper):
        distrib_interval = self.distribution.definition_interval(1e-9)
        selected_lt = self.distribution.cdf(
            upper * np.diff(distrib_interval) + distrib_interval[0]
        )
        selected_gt = self.distribution.sf(
            lower * np.diff(distrib_interval) + distrib_interval[0]
        )
        random_numbers = np.random.rand(n)
        selected = random_numbers <= selected_lt * selected_gt
        return np.count_nonzero(selected)

    def place(self, chunk, indicators):
        for name_indic, indicator in indicators.items():
            all_positions = np.empty((0, 3))
            for p in indicator.partitions:
                num_to_place = indicator.guess(voxels=p.to_voxels())
                partition_size = (p._data.mdc - p._data.ldc)[self.axis]
                chunk_borders = np.array([chunk.ldc[self.axis], chunk.mdc[self.axis]])
                ratios = chunk_borders / partition_size
                if self.direction == "negative":
                    ratios = 1 - ratios
                    ratios = ratios[::-1]

                num_selected = self.draw_interval(num_to_place, *ratios)
                if num_selected > 0:
                    positions = (
                        np.random.rand(num_selected, 3) * chunk.dimensions + chunk.ldc
                    )
                    all_positions = np.concatenate([all_positions, positions])
            self.place_cells(indicator, all_positions, chunk)
