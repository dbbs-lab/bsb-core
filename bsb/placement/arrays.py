from .strategy import PlacementStrategy
import math, numpy as np
from .. import config
from ..config import types
from ..mixins import NotParallel
from ..storage import Chunk
from ..reporting import report, warn


@config.node
class ParallelArrayPlacement(NotParallel, PlacementStrategy):
    """
    Implementation of the placement of cells in parallel arrays.
    """

    spacing_x = config.attr(type=float, required=True)
    angle = config.attr(type=types.deg_to_radian(), required=True)

    def place(self, chunk, indicators):
        """
        Cell placement: Create a lattice of parallel arrays/lines in the layer's surface.
        """
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            radius = indicator.get_radius()
            for prt in self.partitions:
                width, height, depth = prt.data.mdc - prt.data.ldc
                ldc = prt.data.ldc
                # Extension of a single array in the X dimension
                spacing_x = self.spacing_x
                # Add a random shift to the starting points of the arrays for variation.
                x_shift = np.random.rand() * spacing_x
                # Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
                # They are placed in straight lines, tilted by a certain angle by adding a shifting value.
                x_pos = np.arange(start=0.0, stop=width, step=spacing_x) + x_shift
                if x_pos.shape[0] == 0:
                    # When the spacing_x of is larger than the simulation volume,
                    # place a single row on a random position along the x axis
                    x_pos = np.array([x_shift])
                # Amount of parallel arrays of cells
                n_arrays = x_pos.shape[0]
                # Number of cells
                n = np.sum(indicator.guess(prt.data))
                # Add extra cells to fill the lattice error volume which will be pruned
                n += int((n_arrays * spacing_x % width) / width * n)
                # cells to distribute along the rows
                cells_per_row = round(n / n_arrays)
                # The rounded amount of cells that will be placed
                cells_placed = cells_per_row * n_arrays
                # Calculate the position of the cells along the z-axis.
                z_pos, z_axis_distance = np.linspace(
                    start=0.0,
                    stop=depth - radius,
                    num=cells_per_row,
                    retstep=True,
                    endpoint=False,
                )
                # Center the cell soma center to the middle of the unit cell
                z_pos += radius + z_axis_distance / 2
                # The length of the X axis rounded up to a multiple of the unit cell size.
                lattice_x = n_arrays * spacing_x
                # The length of the X axis where cells can be placed in.
                bounded_x = lattice_x - radius * 2
                # Epsilon: open space in the unit cell along the z-axis
                ϵ = z_axis_distance - radius * 2
                # Storage array for the cells
                cells = np.empty((cells_placed, 3))
                for i in range(z_pos.shape[0]):
                    # Shift the arrays at an angle
                    angleShift = z_pos[i] * math.tan(self.angle)
                    # Apply shift and offset
                    x = x_pos + angleShift
                    # Place the cells in a bounded lattice with a little modulus magic
                    x = ldc[0] + x % bounded_x + radius
                    # Place them at a uniformly random height throughout the partition.
                    y = ldc[1] + np.random.uniform(radius, height - radius, x.shape[0])
                    # Place the cells in their z-position with jitter
                    z = ldc[2] + z_pos[i] + ϵ * (np.random.rand(x.shape[0]) - 0.5)
                    # Store this stack's cells
                    cells[(i * len(x)) : ((i + 1) * len(x)), 0] = x
                    cells[(i * len(x)) : ((i + 1) * len(x)), 1] = y
                    cells[(i * len(x)) : ((i + 1) * len(x)), 2] = z
                # Place all the cells in 1 batch (more efficient)
                positions = cells[cells[:, 0] < width - radius]

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
