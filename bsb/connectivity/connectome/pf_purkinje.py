import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomePFPurkinje(ConnectionStrategy):
    """
    Legacy implementation for the connections between parallel fibers and purkinje cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        purkinje_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        first_granule = int(granules[0, 0])
        purkinje_extension_x = purkinje_cell_type.placement.extension_x

        def connectome_pf_pc(first_granule, granules, purkinjes, x_pc):
            pf_pc = np.zeros((0, 2))
            # for all Purkinje cells: calculate and choose which parallel fibers fall into the area of PC dendritic tree (then delete them from successive computations, since 1 parallel fiber is connected to a maximum of PCs)
            for i in purkinjes:
                # which parallel fibers fall into the x range of values?
                bool_matrix = (granules[:, 2]).__ge__(i[2] - x_pc / 2.0) & (
                    granules[:, 2]
                ).__le__(
                    i[2] + x_pc / 2.0
                )  # CAMBIARE IN new_granules SE VINCOLO SU 30 pfs
                good_pf = np.where(bool_matrix)[
                    0
                ]  # finds indexes of parallel fibers that, on the selected axis, satisfy the condition

                # construction of the output matrix: the first column has the GrC id, while the second column has the PC id
                matrix = np.zeros((len(good_pf), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_pf + first_granule
                pf_pc = np.vstack((pf_pc, matrix))

            return pf_pc

        result = connectome_pf_pc(
            first_granule, granules, purkinjes, purkinje_extension_x
        )
        self.scaffold.connect_cells(self, result)
