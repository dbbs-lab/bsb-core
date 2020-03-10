from .strategy import Layered, PlacementStrategy
import math, numpy as np


class ParallelArrayPlacement(Layered, PlacementStrategy):
    """
        Implementation of the placement of cells in parallel arrays.
    """

    casts = {
        "extension_x": float,
        "extension_z": float,
        "angle": lambda x: float(x) * 2 * math.pi / 360,
    }

    defaults = {"angle": 0.08726646259971647}  # 5 degrees

    required = ["extension_x", "extension_z", "angle"]

    def place(self):
        """
            Cell placement: Create a lattice of parallel arrays/lines in the layer's surface.
        """
        cell_type = self.cell_type
        layer = self.layer_instance
        radius = cell_type.placement.radius
        # Extension of a single array in the X dimension
        extension_x = self.extension_x
        # Add a random shift to the starting points of the arrays for variation.
        start_offset = np.random.rand() * extension_x
        # Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
        # They are placed in straight lines, tilted by a certain angle by adding a shifting value.
        x_positions = (
            np.arange(start=0.0, stop=layer.width, step=extension_x) + start_offset
        )
        if (
            x_positions.shape[0] == 0
        ):  # This only happens if the extension_x of a purkinje cell is larger than the simulation volume
            # Place a single row on a random position along the x axis
            x_positions = np.array([start_offset])
        # Amount of parallel arrays of cells
        n_arrays = x_positions.shape[0]
        # Number of cells
        n = self.get_placement_count()
        # Add extra cells to fill the lattice error volume which will be pruned
        n += int((n_arrays * extension_x % layer.width) / layer.width * n)
        # cells to distribute along the rows
        cells_per_row = round(n / n_arrays)
        # The rounded amount of cells that will be placed
        cells_placed = cells_per_row * n_arrays
        # Calculate the position of the cells along the z-axis.
        z_positions, z_axis_distance = np.linspace(
            start=0.0,
            stop=layer.depth - radius,
            num=cells_per_row,
            retstep=True,
            endpoint=False,
        )
        # Center the cell soma center to the middle of the unit cell
        z_positions += radius + z_axis_distance / 2
        # The length of the X axis rounded up to a multiple of the unit cell size.
        lattice_x = n_arrays * extension_x
        # The length of the X axis where cells can be placed in.
        bounded_x = lattice_x - radius * 2
        # Epsilon: jitter along the z-axis
        ϵ = self.extension_z / 2
        # Storage array for the cells
        cells = np.empty((cells_placed, 3))
        for i in np.arange(z_positions.shape[0]):
            # Shift the arrays at an angle
            angleShift = z_positions[i] * math.tan(self.angle)
            # Apply shift and offset
            x = x_positions + angleShift
            # Place the cells in a bounded lattice with a little modulus magic
            x = layer.origin[0] + x % bounded_x + radius
            # Place them at a uniformly random height throughout the layer.
            y = layer.origin[1] + np.random.uniform(
                radius, layer.height - radius, x.shape[0]
            )
            # Place the cells in their z-position with slight jitter
            z = layer.origin[2] + np.array(
                [
                    z_positions[i] + ϵ * (np.random.rand() - 0.5)
                    for _ in np.arange(x.shape[0])
                ]
            )
            # Store this stack's cells
            cells[(i * len(x)) : ((i + 1) * len(x)), 0] = x
            cells[(i * len(x)) : ((i + 1) * len(x)), 1] = y
            cells[(i * len(x)) : ((i + 1) * len(x)), 2] = z
        # Place all the cells in 1 batch (more efficient)
        self.scaffold.place_cells(
            cell_type, layer, cells[cells[:, 0] < layer.width - radius]
        )
