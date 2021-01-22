import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomePFInterneuron(ConnectionStrategy):
    """
    Legacy implementation for the connections between parallel fibers and a molecular layer interneuron cell_type.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        interneuron_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        interneurons = self.scaffold.cells_by_type[interneuron_cell_type.name]
        first_granule = int(granules[0, 0])
        dendrite_radius = interneuron_cell_type.morphology.dendrite_radius
        # Spoof fixed pf height of 150 µm
        pf_heights = 150 + granules[:, 3]

        def connectome_pf_inter(first_granule, interneurons, granules, r_sb, h_pf):
            pf_interneuron = np.zeros((0, 2))
            # for each interneuron find all the parallel fibers that fall into the sphere with centre the cell soma and appropriate radius
            for i in interneurons:

                # find all cells that satisfy the condition
                interneuron_matrix = (
                    ((granules[:, 2] - i[2]) ** 2) + ((h_pf - i[3]) ** 2) - (r_sb ** 2)
                ).__le__(0)
                # indexes of interneurons that can potentially be connected
                good_pf = np.where(interneuron_matrix)[0]

                matrix = np.zeros((len(good_pf), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_pf + first_granule
                pf_interneuron = np.vstack((pf_interneuron, matrix))

            return pf_interneuron

        result = connectome_pf_inter(
            first_granule, interneurons, granules, dendrite_radius, pf_heights
        )
        self.scaffold.connect_cells(self, result)
