import numpy as np
import csv
from ..strategy import ConnectionStrategy
from ...reporting import report, warn


class ConnectomeDcnGlyGolgi(ConnectionStrategy):
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
        convergence = int(self.convergence)

        # GCL geometries
        y_min = self.scaffold.configuration.layers["dcn_layer"].dimensions[
            1
        ]  # this value represents the beginning of GCL along y axis
        BoundsX = np.array(
            [0, self.scaffold.configuration.layers["granular_layer"].dimensions[0]]
        )
        BoundsY = np.array(
            [
                y_min,
                y_min
                + self.scaffold.configuration.layers["granular_layer"].dimensions[1],
            ]
        )
        BoundsZ = np.array(
            [0, self.scaffold.configuration.layers["granular_layer"].dimensions[2]]
        )
        # print(BoundsX)
        # print(BoundsY)
        # print(BoundsZ)

        # Connections are created until unconnected_dcn is empty
        unconnected_dcn = dcn_cells
        Y_centers = []
        Z_centers = []
        intercluster_distance = 0  # initialization value that allows to enter in the while loop (line 55) when constructing the second cluster
        pre_post = np.zeros([0, 2])
        test = 0
        y_intracluster_dist = 10
        z_intracluster_dist = 10
        Y = []
        Z = []
        connected_golgi = np.zeros((0, 5))
        selected_golgi = golgi_cells
        while len(unconnected_dcn[:, 0]) > 0:
            # print("Numero di cellule da connettere:", len(unconnected_dcn[:, 0]))
            # print("Cellule da connettere:", unconnected_dcn[:, 0])
            # print("Golgi considerate:", len(selected_golgi))
            cluster_numerosity = convergence
            cluster_numerosity = np.min(
                [cluster_numerosity, len(unconnected_dcn[:, 0])]
            )  # last cycle len(unconnected)<cluster_numerosity
            actual_cluster = np.random.choice(
                len(unconnected_dcn[:, 0]), cluster_numerosity, replace=False
            )
            # print("righe selezionate:", actual_cluster)
            dcn_selected = unconnected_dcn[actual_cluster, :]
            # print("dcn selezionate:", dcn_selected[:, 0])
            unconnected_dcn = np.delete(
                unconnected_dcn, actual_cluster, axis=0
            )  # Update unconnected_dcn for next cycle

            # Select a random point in the X-Z plane which represents
            # the center of the cluster of dcn-selected
            if len(Y_centers) > 0 or len(Z_centers) > 0:
                while (
                    np.min(intercluster_distance) < 70
                ):  # it assures at least 50um distance between two axons in two different clusters
                    Y_center_cluster = np.random.randint(
                        BoundsY[0] + y_intracluster_dist,
                        BoundsY[1] - y_intracluster_dist,
                        size=1,
                    )
                    Z_center_cluster = np.random.randint(
                        BoundsZ[0] + z_intracluster_dist,
                        BoundsZ[1] - z_intracluster_dist,
                        size=1,
                    )
                    intercluster_distance = [
                        np.sqrt(
                            (Y_centers[i] - Y_center_cluster) ** 2
                            + (Z_centers[i] - Z_center_cluster) ** 2
                        )
                        for i in range(np.size(Y_centers))
                    ]
                    intercluster_distance = np.array(intercluster_distance)
                    # print("valore minimo:", np.min(intercluster_distance))
                    test += 1
                intercluster_distance = 0
            else:
                test += 1
                Y_center_cluster = np.random.randint(
                    BoundsY[0] + y_intracluster_dist,
                    BoundsY[1] - y_intracluster_dist,
                    size=1,
                )
                Z_center_cluster = np.random.randint(
                    BoundsZ[0] + z_intracluster_dist,
                    BoundsZ[1] - z_intracluster_dist,
                    size=1,
                )

            Y_centers = np.append(Y_centers, Y_center_cluster)
            Z_centers = np.append(Z_centers, Z_center_cluster)

            # Definition of a 50x50 square region around the center of the cluster
            BoundsY_cluster = [
                Y_center_cluster - y_intracluster_dist,
                Y_center_cluster + y_intracluster_dist,
            ]
            BoundsZ_cluster = [
                Z_center_cluster - z_intracluster_dist,
                Z_center_cluster + z_intracluster_dist,
            ]
            YZ_Area = (BoundsY_cluster[1] - BoundsY_cluster[0]) * (
                BoundsZ_cluster[1] - BoundsZ_cluster[0]
            )

            # Create a uniform distribution of DCN cells in the square region (Y-Z)
            n_dcn = len(dcn_selected[:, 0])
            dcn_per_Area = n_dcn / YZ_Area
            dcn_per_Y = int(
                np.ceil((BoundsY_cluster[1] - BoundsY_cluster[0]) * np.sqrt(dcn_per_Area))
            )
            dcn_per_Z = int(
                np.ceil((BoundsZ_cluster[1] - BoundsZ_cluster[0]) * np.sqrt(dcn_per_Area))
            )
            dcn_Y = np.linspace(BoundsY_cluster[0], BoundsY_cluster[1], num=dcn_per_Y)
            dcn_Z = np.linspace(BoundsZ_cluster[0], BoundsZ_cluster[1], num=dcn_per_Z)
            yv, zv = np.meshgrid(dcn_Y, dcn_Z, sparse=False, indexing="ij")
            yv = yv.flatten()
            zv = zv.flatten()

            # Limit the number of DCN cells (yv and zv) to n_dcn
            if np.size(yv) > n_dcn:
                delete_points = np.random.choice(
                    np.size(yv), size=np.size(yv) - n_dcn, replace=False
                )
                yv = np.delete(yv, delete_points)
                zv = np.delete(zv, delete_points)

            Y.append(yv)
            Z.append(zv)

            distance_array = np.zeros(len(selected_golgi))
            for j in range(len(selected_golgi)):
                for i in range(len(yv)):
                    projected_point = np.array([selected_golgi[j, 2], yv[i], zv[i]])
                    distance_array[j] += np.sqrt(
                        (projected_point[0] - selected_golgi[j, 2]) ** 2
                        + (projected_point[1] - selected_golgi[j, 3]) ** 2
                        + (projected_point[2] - selected_golgi[j, 4]) ** 2
                    )
            sorted_distance_indexes = np.argsort(
                distance_array
            )  # sorted so that the first values correspond to the closest golgi cells
            divergence = int(self.divergence)
            best_golgi_cells = selected_golgi[sorted_distance_indexes[0:divergence], :]
            connected_golgi = np.vstack((connected_golgi, best_golgi_cells))
            selected_golgi = np.delete(
                selected_golgi, sorted_distance_indexes[0:divergence], axis=0
            )  # Update golgi cells for next cycle
            connections = np.zeros([divergence, 2])

            for k in range(len(yv)):
                connections[:, 0] = dcn_selected[k, 0]
                connections[:, 1] = best_golgi_cells[:, 0]
                pre_post = np.vstack((pre_post, connections))
        connected_golgi = connected_golgi[np.argsort(connected_golgi[:, 0]), :]

        self.scaffold.connect_cells(self, pre_post)
