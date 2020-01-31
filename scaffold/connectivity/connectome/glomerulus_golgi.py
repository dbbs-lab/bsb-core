import numpy as np
from ..strategy import TouchingConvergenceDivergence


class ConnectomeGlomerulusGolgi(TouchingConvergenceDivergence):
    """
        Legacy implementation for the connections between Golgi cells and glomeruli.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        glomerulus_cell_type = self.from_cell_types[0]
        golgi_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_glomerulus = int(glomeruli[0, 0])
        r_goc_vol = golgi_cell_type.morphology.dendrite_radius

        def connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol):
            glom_bd = np.zeros((0, 2))
            glom_x = glomeruli[:, 2]
            glom_y = glomeruli[:, 3]
            glom_z = glomeruli[:, 4]

            # for all Golgi cells: calculate which glomeruli fall into the volume of GoC basolateral dendrites, then choose 40 of them for the connection and delete them from successive computations, since 1 axon is connected to 1 GoC
            for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in golgicells:
                # Geometric constraints: glom less than `r_goc_vol` away from golgi and golgi cell soma above glom.
                volume_matrix = (
                    ((glom_x - golgi_x) ** 2)
                    + ((glom_y - golgi_y) ** 2)
                    + ((glom_z - golgi_z) ** 2)
                    - (r_goc_vol ** 2)
                ).__le__(0) & glom_y.__le__(golgi_y)
                good_gloms = np.where(volume_matrix == True)[
                    0
                ]  # finds indexes of granules that can potentially be connected
                connected_gloms = (
                    good_gloms + first_glomerulus
                )  # Translate local id to simulation id

                matrix = np.zeros((len(good_gloms), 2))
                matrix[:, 0] = connected_gloms  # from cell
                matrix[:, 1] = golgi_id  # to cell
                glom_bd = np.vstack((glom_bd, matrix))

            return glom_bd

        connectome = connectome_glom_goc(first_glomerulus, glomeruli, golgis, r_goc_vol)
        self.scaffold.connect_cells(self, connectome)
