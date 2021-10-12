import numpy as np
from ..strategy import ConnectionStrategy
from ...reporting import report, warn


class ConnectomeDcnGranule(ConnectionStrategy):
    """
    Implementation for the connections between mossy fibers and glomeruli.
    The connectivity is somatotopic and
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        dcn_cell_type = self.from_cell_types[0]
        granule_cell_type = self.to_cell_types[0]
        dcn_cells = self.from_cells[dcn_cell_type.name]
        granule_cells = self.to_cells[granule_cell_type.name]
        divergence = int(self.divergence)

        # NC fibers reach only superficially the granule cell layer: only 50 um
        y_min = self.scaffold.configuration.layers["dcn_layer"].dimensions[
            1
        ]  # this value represents the beginning of GCL
        thickness_y = 50  # only the superficial part of GCL
        granule_selected = granule_cells[
            np.where(granule_cells[:, 3] <= y_min + thickness_y)[0], :
        ]
        N_MF = len(dcn_cells[:, 0])  # Each NC DCN cell forms 1 MF

        # Position of phantom glom are taken from the ConnectomeDcnGolgi
        connettoma_DCN_golgi = self.scaffold.configuration.connection_types[
            "NC_dcn_glut_large_to_golgi"
        ]
        x_points = connettoma_DCN_golgi.x_points
        y_points = connettoma_DCN_golgi.y_points
        z_points = connettoma_DCN_golgi.z_points

        pre_post = np.zeros((divergence * N_MF, 2))
        k = 0
        for i in range(N_MF):
            distance_array = np.zeros(len(granule_selected))
            for j in range(len(granule_selected)):
                distance_array[j] = np.sqrt(
                    (x_points[i] - granule_selected[j, 2]) ** 2
                    + (y_points[i] - granule_selected[j, 3]) ** 2
                    + (z_points[i] - granule_selected[j, 4]) ** 2
                )
            sorted_distance_indexes = np.argsort(
                distance_array
            )  # sorted so that the first values correspond to the closest granule cells
            best_granule = granule_selected[sorted_distance_indexes[0:divergence], 0]
            pre_post[divergence * k : divergence * (k + 1), 0] = dcn_cells[i, 0]
            pre_post[divergence * k : divergence * (k + 1), 1] = best_granule
            k += 1

        self.scaffold.connect_cells(self, pre_post)
