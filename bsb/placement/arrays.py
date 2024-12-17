import math

import numpy as np

from .. import config
from ..config import types
from ..exceptions import ConfigurationError, PackingError
from ..mixins import NotParallel
from ..reporting import report
from .strategy import PlacementStrategy


@config.node
class ParallelArrayPlacement(NotParallel, PlacementStrategy):
    """
    Implementation of the placement of cells in parallel arrays
    Cells are placed in rows on the plane defined by 2 selected axes.
    """

    spacing_x: float = config.attr(type=types.float(min=0), required=True)
    """Space in between two cells along the main axis"""
    angle: float = config.attr(type=types.deg_to_radian(), required=True)
    """Angle between the second axis and the axis of the rows of cells"""

    def boot(self):
        if self.angle % (np.pi) == np.pi / 2:
            raise ConfigurationError(
                f"Parallel array angle should be not a multiple of pi/2 for '{self.name}'. Provided angle: {self.angle}"
            )

    def place(self, chunk, indicators):
        """
        Cell placement: Create a lattice of parallel arrays/lines in the (x, y) surface.
        """
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            radius = indicator.get_radius()
            for prt in self.partitions:
                width, depth, height = prt.data.mdc - prt.data.ldc
                ldc = prt.data.ldc
                # Add a random shift to the starting points of the arrays for variation.
                x_shift = np.random.rand() * self.spacing_x
                # Place cells equally spaced over the entire length of the X axis kept apart by the provided space.
                # They are placed in straight lines, tilted by a certain angle by adding a shifting value.
                x_pos = np.arange(start=0.0, stop=width, step=self.spacing_x) + x_shift
                if x_pos.shape[0] == 0:
                    # When the spacing_x of is larger than the simulation volume,
                    # place a single row on a random position along the X axis
                    x_pos = np.array([x_shift])
                # Amount of parallel arrays of cells
                n_arrays = x_pos.shape[0]
                # Number of cells
                n = np.sum(indicator.guess(prt.data))
                # Add extra cells to fill the lattice error volume which will be pruned
                n += int((n_arrays * self.spacing_x % width) / width * n)
                # cells to distribute along the rows
                cells_per_row = round(n / n_arrays)
                # The rounded amount of cells that will be placed
                cells_placed = cells_per_row * n_arrays
                # Calculate the position of the cells along the z-axis.
                y_pos, y_axis_distance = np.linspace(
                    start=0.0,
                    stop=depth - radius,
                    num=cells_per_row,
                    retstep=True,
                    endpoint=False,
                )
                # Center the cell soma center to the middle of the unit cell
                y_pos += radius + y_axis_distance / 2
                # The length of the X axis rounded up to a multiple of the unit cell size.
                lattice_x = n_arrays * self.spacing_x
                # The length of the X axis where cells can be placed in.
                bounded_x = lattice_x - radius * 2
                # Epsilon: open space in the unit cell along the Y axis
                epsilon = y_axis_distance / math.cos(self.angle) - radius * 2
                if epsilon < 0:
                    raise PackingError(
                        f"Not enough space between cells placed on the same row for '{self.name}'."
                    )
                # Storage array for the cells
                cells = np.empty((cells_placed, 3))
                for i in range(y_pos.shape[0]):
                    # Shift the arrays at an angle
                    angleShift = y_pos[i] * math.tan(self.angle)
                    # Apply shift and offset
                    x = x_pos + angleShift
                    # Place the cells in a bounded lattice with a little modulus magic
                    x = ldc[0] + x % bounded_x + radius
                    # Place the cells in their y-position with jitter
                    y = (
                        ldc[1]
                        + y_pos[i]
                        + epsilon
                        * (np.random.rand(x.shape[0]) - 0.5)
                        * math.cos(self.angle)
                    )
                    # Place them at a uniformly random height throughout the partition.
                    z = ldc[2] + np.random.uniform(radius, height - radius, x.shape[0])
                    # Store this stack's cells
                    cells[(i * len(x)) : ((i + 1) * len(x)), 0] = x
                    cells[(i * len(x)) : ((i + 1) * len(x)), 1] = y
                    cells[(i * len(x)) : ((i + 1) * len(x)), 2] = z
                # Place all the cells in 1 batch (more efficient)
                positions = cells[cells[:, 0] < prt.data.ldc[0] + width - radius]

                # Determine in which chunks the cells must be placed
                cs = self.scaffold.configuration.network.chunk_size
                chunks_list = np.array(
                    [chunk.data + np.floor_divide(p, cs[0]) for p in positions]
                )
                unique_chunks_list = np.unique(chunks_list, axis=0)

                # For each chunk, place the cells
                for c in unique_chunks_list:
                    idx = np.where((chunks_list == c).all(axis=1))
                    pos_current_chunk = positions[idx]
                    self.place_cells(indicator, pos_current_chunk, chunk=c)
                report(f"Placed {len(positions)} {cell_type.name} in {prt.name}", level=3)


__all__ = ["ParallelArrayPlacement"]
