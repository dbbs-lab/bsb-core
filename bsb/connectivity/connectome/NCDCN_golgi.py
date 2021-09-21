import numpy as np
from ..strategy import ConnectionStrategy
from ...reporting import report, warn


class ConnectomeDcnGolgi(ConnectionStrategy):
    """
    Implementation for the connections between mossy fibers and glomeruli.
    The connectivity is somatotopic and
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        dcn_cell_type = self.from_cell_types[0]
        golgi_cell_type = self.to_cell_types[0]
        dcn_cells = self.from_cells[dcn_cell_type.name]
        golgi_cells = self.to_cells[golgi_cell_type.name]
        divergence = int(self.divergence)

        # NC fibers reach only superficially the granule cell layer: only the first 50 um
        y_min = self.scaffold.configuration.layers["dcn_layer"].dimensions[
            1
        ]  # this value represents the beginning of GCL
        thickness_y = 50  # only the superficial part of GCL
        golgi_selected = golgi_cells[
            np.where(golgi_cells[:, 3] <= y_min + thickness_y)[0], :
        ]

        # Boundaries of X, Y and Z space for Golgi
        BoundsX = np.array(
            [0, self.scaffold.configuration.layers["granular_layer"].dimensions[0]]
        )
        BoundsY = np.array([y_min, y_min + thickness_y])
        BoundsZ = np.array(
            [0, self.scaffold.configuration.layers["granular_layer"].dimensions[2]]
        )

        # Density dcn NC mossy fibers
        vol_xyz = (
            (BoundsX[1] - BoundsX[0])
            * (BoundsY[1] - BoundsY[0])
            * (BoundsZ[1] - BoundsZ[0])
        )
        N_MF = len(dcn_cells[:, 0])  # Each NC DCN cell forms 1 MF
        dcn_glom_density = (
            N_MF / vol_xyz
        )  # Each NC mossy fiber forms just 1 rosette-like terminal (phantom glom)

        # Numerosity along the X, Z axes
        MF_per_X = np.ceil(
            (BoundsX[1] - BoundsX[0]) * np.power(dcn_glom_density, 1 / 3)
        ).astype(int)
        MF_per_Z = np.ceil(
            (BoundsZ[1] - BoundsZ[0]) * np.power(dcn_glom_density, 1 / 3)
        ).astype(int)

        # Points where virtual rosettes are present
        delta_x = (BoundsX[1] - BoundsX[0]) / MF_per_X
        delta_y = (
            BoundsY[1] - BoundsY[0]
        ) / MF_per_X  # MF_per_X in order to have an oblique or random distribution along y
        delta_z = (BoundsZ[1] - BoundsZ[0]) / MF_per_Z

        # Points start not from the edges of the volume
        MF_X = np.arange(BoundsX[0] + delta_x / 2, BoundsX[1], delta_x)
        MF_Y = np.arange(BoundsY[0] + delta_y / 2, BoundsY[1], delta_y)
        MF_Z = np.arange(BoundsZ[0] + delta_z / 2, BoundsZ[1], delta_z)

        # Construct the X,Z grid of points
        self.x_points, self.z_points = np.meshgrid(
            MF_X, MF_Z, sparse=False, indexing="ij"
        )
        self.x_points = self.x_points.flatten()
        self.z_points = self.z_points.flatten()

        # For each point in the column(i) of points -> [X = Xi and Z = [Zo:Zn]]
        # we assign a different y value taken from MF_Y
        np.random.shuffle(MF_Y)
        self.y_points = np.array(MF_Y)
        for i in range(MF_per_X - 1):
            np.random.shuffle(MF_Y)
            self.y_points = np.hstack((self.y_points, MF_Y))

        # The number of points could be higher than the number of NC MF
        if len(self.x_points) > N_MF:
            delete_points = np.random.choice(
                len(self.x_points), len(self.x_points) - N_MF, replace=False
            )
            self.x_points = np.delete(self.x_points, delete_points)
            self.y_points = np.delete(self.y_points, delete_points)
            self.z_points = np.delete(self.z_points, delete_points)

        # Connectome construction based on minimum distance phantom glom <-> Golgi cell
        pre_post = np.zeros((divergence * N_MF, 2))
        k = 0
        for i in range(N_MF):
            distance_array = np.zeros(len(golgi_selected))
            for j in range(len(golgi_selected)):
                distance_array[j] = np.sqrt(
                    (self.x_points[i] - golgi_selected[j, 2]) ** 2
                    + (self.y_points[i] - golgi_selected[j, 3]) ** 2
                    + (self.z_points[i] - golgi_selected[j, 4]) ** 2
                )
            sorted_distance_indexes = np.argsort(
                distance_array
            )  # sorted so that the first values correspond to the closest golgi cells
            best_golgi = golgi_selected[sorted_distance_indexes[0:divergence], 0]
            pre_post[divergence * k : divergence * (k + 1), 0] = dcn_cells[i, 0]
            pre_post[divergence * k : divergence * (k + 1), 1] = best_golgi
            k += 1

        self.scaffold.connect_cells(self, pre_post)
