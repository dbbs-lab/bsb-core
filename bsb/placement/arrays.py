from .strategy import PlacementStrategy
import math, numpy as np
from .. import config
from ..config import types
from ..reporting import report, warn


@config.node
class ParallelArrayPlacement(PlacementStrategy):
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
                x_pos = np.arange(start=0.0, stop=width, step=spacing_x)
                if x_pos.shape[0] == 0:
                    # When the spacing_x of is larger than the simulation volume,
                    # place a single row on a random position along the x axis
                    x_pos = np.array([x_shift])
                #print(prt.data.mdc)
                zk = int(prt.data.mdc[2] // (2*int(radius)))
                shift_x = int(int(prt.data.mdc[0]) % int(2*radius))
                # Number of cells in chunk
                #n = np.sum(indicator.guess(chunk))
                deltas = 5*radius*(np.random.uniform())
                max_cell_per_array = int(prt.data.mdc[0]/radius) 
                cells_placed = zk*max_cell_per_array
                cells = np.empty((cells_placed, 3))             

                ptr = 0
                for i in range(max_cell_per_array):
                    for j in range(zk):
                        #Generate (little) random shifts in x and z position
                        rng1 = 0.25*radius*np.random.uniform()
                        rng2 = 0.5*radius*np.random.uniform()

                        x = spacing_x*i+j*2*radius + rng1
                        z = np.tan(self.angle)*(j*2*radius + rng2)
                        y = ldc[1] + np.random.uniform(radius, height - radius)
                        cells[ptr,0] = x
                        cells[ptr,1] = y
                        cells[ptr,2] = z
                        ptr += 1

                ok_cells = (cells[:,0] <= (chunk[0]+1)*chunk.dimensions[0]) & (cells[:,0] >= (chunk[0])*chunk.dimensions[0])
                cells = cells[ok_cells]
                ok_cells = (cells[:,2] <= (chunk[2]+1)*chunk.dimensions[2]) & (cells[:,2] >= (chunk[2])*chunk.dimensions[2])
                positions = cells[ok_cells]

                print(positions)
                """from scipy.spatial.transform import Rotation as R
                rot = R.from_euler('y',+70, degrees=True)
                indicator.rotations = rot"""

                report(f"Placing {len(positions)} {cell_type.name} in {chunk}", level=3)
                self.place_cells(indicator, positions, chunk=chunk)


    """
    ORIGINAL IMPLEMENTATION OF PARALLEL ARRAY STRATEGY

    spacing_x = config.attr(type=float, required=True)
    angle = config.attr(type=types.deg_to_radian(), required=True)

    def place(self, chunk, indicators):

        for indicator in indicators.values():
            cell_type = indicator.cell_type
            radius = indicator.get_radius()
            for prt in self.partitions:
                width, height, depth = prt.data.mdc - prt.data.ldc
                ldc = prt.data.ldc
                # Extension of a single array in the X dimension
                spacing_x = self.spacing_x
                print(spacing_x)
                # Add a random shift to the starting points of the arrays for variation.
                x_shift = np.random.rand() * spacing_x
                # Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
                # They are placed in straight lines, tilted by a certain angle by adding a shifting value.
                x_pos = np.arange(start=0.0, stop=width, step=spacing_x) + x_shift
                if x_pos.shape[0] == 0:
                    # When the spacing_x of is larger than the simulation volume,
                    # place a single row on a random position along the x axis
                    x_pos = np.array([x_shift])
                #Total amount of parallel arrays of cells
                n_arrays = x_pos.shape[0]
                # Number of cells
                #print(chunk.dimensions)
                #print(prt.to_chunks(chunk.dimensions))
                #print(indicator.guess(chunk))
                n = np.sum(indicator.guess(chunk))#*len(prt.to_chunks(chunk.dimensions))
                # Add extra cells to fill the lattice error volume which will be pruned
                n += int((n_arrays * spacing_x % width) / width * n)
                print("Expected total number",n)
                #print(n_arrays)
                # cells to distribute along the rows
                cells_per_row = int(np.floor(n / n_arrays))
                print("Number of cells per row:", cells_per_row)
                # The rounded amount of cells that will be placed
                cells_placed = cells_per_row * n_arrays
                
                ps = indicator.cell_type.get_placement_set().load_positions()
                if (ps.size != 0):
                    break
                    print(ps)
                    mdist = np.min(np.abs(chunk[0]-ps[:,0]))
                    print(mdist)
                    if (mdist > spacing_x):
                        cells_placed = 0
                        cells_per_row = 0
                print(cells_placed)
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
                #print(len(positions))
                report(f"Placing {len(positions)} {cell_type.name} in {chunk}", level=3)
                ct = indicator.cell_type
                pos_p = ct.get_placement_set().load_positions()
                print("Total number of Purkinje:", len(pos_p))
                self.place_cells(indicator, positions, chunk=chunk)


    """