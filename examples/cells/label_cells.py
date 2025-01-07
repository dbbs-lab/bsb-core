# Create an After placement hook that will label cells according to their position
import numpy as np

from bsb import AfterPlacementHook, config, refs, types


class LabelCellA(AfterPlacementHook):
    """
    Subdivide a cell population into 2 subpopulations based on their
    position along a provided axis
    """

    cell_type: str = config.ref(refs.cell_type_ref, required=True)
    """Reference to the cell type."""

    axis: int = config.attr(type=types.int(min=0, max=2), default=0)
    """Axis along which to subdivide the population."""

    def postprocess(self):
        # Load the cell type positions
        ps = self.scaffold.get_placement_set(self.cell_type)
        cell_positions = ps.load_positions()

        # create a filter that split the cells according to
        # the mean of their positions along the chosen axis
        mean_pos = np.mean(cell_positions[:, self.axis])
        subpopulation_1 = np.asarray(cell_positions >= mean_pos, dtype=int)
        subpopulation_2 = np.asarray(cell_positions < mean_pos, dtype=int)

        # set the cell label according to the filter
        ps.label(labels=["cell_A_type_1"], cells=subpopulation_1)
        ps.label(labels=["cell_A_type_2"], cells=subpopulation_2)
