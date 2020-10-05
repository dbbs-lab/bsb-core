import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomePurkinjeDCN(ConnectionStrategy):
    """
    Legacy implementation for the connection between purkinje cells and DCN cells.
    Also rotates the dendritic trees of the DCN.
    """

    casts = {"divergence": int}

    required = ["divergence"]

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        purkinjes = self.from_cells[from_type.name]
        dcn_cells = self.to_cells[to_type.name]

        dend_tree_coeff = np.zeros((dcn_cells.shape[0], 4))
        for i in range(len(dcn_cells)):
            # Make the planar coefficients a, b and c.
            dend_tree_coeff[i] = np.random.rand(4) * 2.0 - 1.0
            # Calculate the last planar coefficient d from ax + by + cz - d = 0
            # => d = - (ax + by + cz)
            dend_tree_coeff[i, 3] = -np.sum(dend_tree_coeff[i, 0:2] * dcn_cells[i, 2:4])

        if len(dcn_cells) == 0:
            return

        first_dcn = int(dcn_cells[0, 0])
        divergence = self.divergence

        def connectome_pc_dcn(first_dcn, purkinjes, dcn_cells, div_pc, dend_tree_coeff):
            pc_dcn = np.zeros((0, 2))

            # For all Purkinje cells: calculate the distance with the area around glutamatergic DCN cells soma, then choose 4-5 of them
            for i in purkinjes:
                distance = (
                    np.absolute(
                        (dend_tree_coeff[:, 0] * i[2])
                        + (dend_tree_coeff[:, 1] * i[3])
                        + (dend_tree_coeff[:, 2] * i[4])
                        + dend_tree_coeff[:, 3]
                    )
                ) / (
                    np.sqrt(
                        (dend_tree_coeff[:, 0] ** 2)
                        + (dend_tree_coeff[:, 1] ** 2)
                        + (dend_tree_coeff[:, 2] ** 2)
                    )
                )

                dist_matrix = np.zeros((dcn_cells.shape[0], 2))
                dist_matrix[:, 1] = dcn_cells[:, 0]
                dist_matrix[:, 0] = distance
                dcn_dist = np.random.permutation(dist_matrix)

                # If the number of DCN cells are less than the divergence value, all neurons are connected to the corresponding PC
                if dcn_cells.shape[0] < div_pc:
                    matrix = np.zeros((dcn_cells.shape[0], 2))
                    matrix[:, 0] = i[0]
                    matrix[:, 1] = dcn_cells[:, 0]
                    pc_dcn = np.vstack((pc_dcn, matrix))

                else:
                    if np.random.rand() > 0.5:

                        connected_f = dcn_dist[0:div_pc, 1]
                        connected_dist = dcn_dist[0:div_pc, 0]
                        connected_provv = connected_f.astype(int)
                        connected_dcn = connected_provv

                        # construction of the output matrix: the first column has the  PC index, while the second column has the connected DCN cell index
                        matrix = np.zeros((div_pc, 2))
                        matrix[:, 0] = i[0]
                        matrix[:, 1] = connected_dcn
                        pc_dcn = np.vstack((pc_dcn, matrix))

                    else:
                        connected_f = dcn_dist[0 : (div_pc - 1), 1]
                        connected_dist = dcn_dist[0 : (div_pc - 1), 0]
                        connected_provv = connected_f.astype(int)
                        connected_dcn = connected_provv

                        matrix = np.zeros(((div_pc - 1), 2))
                        matrix[:, 0] = i[0]
                        matrix[:, 1] = connected_dcn
                        pc_dcn = np.vstack((pc_dcn, matrix))
            return pc_dcn

        results = connectome_pc_dcn(
            first_dcn, purkinjes, dcn_cells, divergence, dend_tree_coeff
        )
        self.scaffold.connect_cells(self, results)
