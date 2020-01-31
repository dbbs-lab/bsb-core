import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeAscAxonPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connections between ascending axons and purkinje cells.
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
        OoB_value = (
            self.scaffold.configuration.X * 1000.0
        )  # Any arbitrarily large value outside of simulation volume
        purkinje_extension_x = purkinje_cell_type.placement.extension_x
        purkinje_extension_z = purkinje_cell_type.placement.extension_z

        def connectome_aa_pc(first_granule, granules, purkinjes, x_pc, z_pc, OoB_value):
            aa_pc = np.zeros((0, 2))
            new_granules = np.copy(granules)

            # for all Purkinje cells: calculate and choose which granules fall into the area of PC dendritic tree, then delete them from successive computations, since 1 ascending axon is connected to only 1 PC
            for i in purkinjes:
                # ascending axon falls into the z range of values?
                bool_vector = (new_granules[:, 4]).__ge__(i[4] - z_pc / 2.0) & (
                    new_granules[:, 4]
                ).__le__(i[4] + z_pc / 2.0)
                # ascending axon falls into the x range of values?
                bool_vector = bool_vector & (
                    (new_granules[:, 2]).__ge__(i[2] - x_pc / 2.0)
                    & (new_granules[:, 2]).__le__(i[2] + x_pc / 2.0)
                )
                good_aa = np.where(bool_vector)[
                    0
                ]  # finds indexes of ascending axons that, on the selected axis, have the correct sum value

                # construction of the output matrix: the first column has the GrC id, while the second column has the PC id
                matrix = np.zeros((len(good_aa), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_aa + first_granule
                aa_pc = np.vstack((aa_pc, matrix))

                new_granules[
                    good_aa, :
                ] = OoB_value  # update the granules matrix used for computation by deleting the coordinates of connected ones

            return aa_pc

        result = connectome_aa_pc(
            first_granule,
            granules,
            purkinjes,
            purkinje_extension_x,
            purkinje_extension_z,
            OoB_value,
        )
        self.scaffold.connect_cells(self, result)
