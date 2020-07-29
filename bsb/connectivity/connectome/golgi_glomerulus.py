import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeGolgiGlomerulus(ConnectionStrategy):
    """
        Legacy implementation for the connections between glomeruli and Golgi cells.
    """

    casts = {"divergence": int}
    required = ["divergence"]

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        golgi_cell_type = self.from_cell_types[0]
        glomerulus_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_glomerulus = int(glomeruli[0, 0])
        GoCaxon_x = golgi_cell_type.morphology.axon_x
        GoCaxon_y = golgi_cell_type.morphology.axon_y
        GoCaxon_z = golgi_cell_type.morphology.axon_z
        r_glom = glomerulus_cell_type.placement.radius
        n_conn_goc = self.divergence
        layer_thickness = self.scaffold.configuration.get_layer(
            name=golgi_cell_type.placement.layer
        ).thickness
        # An arbitrarily large value that will be used to exclude cells from geometric constraints
        oob = self.scaffold.configuration.X * 1000.0

        def connectome_goc_glom(
            first_glomerulus,
            glomeruli,
            golgicells,
            GoCaxon_x,
            GoCaxon_y,
            GoCaxon_z,
            r_glom,
            n_conn_goc,
            layer_thickness,
            oob,
        ):
            glom_x = glomeruli[:, 2]
            glom_y = glomeruli[:, 3]
            glom_z = glomeruli[:, 4]
            new_glomeruli = np.copy(glomeruli)
            new_golgicells = np.random.permutation(golgicells)
            connections = np.zeros((golgis.shape[0] * n_conn_goc, 2))
            new_connection_index = 0

            # for all Golgi cells: calculate which glomeruli fall into the area of GoC
            # axon, then choose 40 of them for the connection and delete them from
            # successive computations, since 1 glomerulus must be connected to only 1 GoC
            for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in new_golgicells:
                # Check geometrical constraints
                # glomerulus falls into the x range of values?
                bool_vector = ((glom_x + r_glom).__ge__(golgi_x - GoCaxon_x / 2.0)) & (
                    (glom_x - r_glom).__le__(golgi_x + GoCaxon_x / 2.0)
                )
                # glomerulus falls into the y range of values?
                bool_vector = bool_vector & (
                    ((glom_y + r_glom).__ge__(golgi_y - GoCaxon_y / 2.0))
                    & ((glom_y - r_glom).__le__(golgi_y + GoCaxon_y / 2.0))
                )
                # glomerulus falls into the z range of values?
                bool_vector = bool_vector & (
                    ((glom_z + r_glom).__ge__(golgi_z - GoCaxon_z / 2.0))
                    & ((glom_z - r_glom).__le__(golgi_z + GoCaxon_z / 2.0))
                )

                # Make a permutation of all candidate glomeruli
                good_gloms = np.where(bool_vector)[0]
                chosen_rand = np.random.permutation(good_gloms)
                good_gloms_matrix = new_glomeruli[chosen_rand]
                # Calculate the distance between the golgi cell and all glomerulus candidates, normalize distance by layer thickness
                normalized_distance_vector = (
                    np.sqrt(
                        (good_gloms_matrix[:, 2] - golgi_x) ** 2
                        + (good_gloms_matrix[:, 3] - golgi_y) ** 2
                    )
                    / layer_thickness
                )
                sorting_map = normalized_distance_vector.argsort()
                # Sort the candidate glomerulus matrix and distance vector by the distance vector
                good_gloms_matrix = good_gloms_matrix[sorting_map]
                # Use the normalized distance vector as a probability treshold for connecting glomeruli
                probability_treshold = normalized_distance_vector[sorting_map]

                idx = 1
                for candidate_index, glomerulus in enumerate(good_gloms_matrix):
                    if idx <= n_conn_goc:
                        ra = np.random.random()
                        if ra.__gt__(probability_treshold[candidate_index]):
                            glomerulus_id = glomerulus[0]
                            connections[new_connection_index, 0] = golgi_id
                            connections[new_connection_index, 1] = glomerulus_id
                            new_glomeruli[int(glomerulus_id - first_glomerulus), :] = oob
                            new_connection_index += 1
                            idx += 1
            return connections[0:new_connection_index]

        result = connectome_goc_glom(
            first_glomerulus,
            glomeruli,
            golgis,
            GoCaxon_x,
            GoCaxon_y,
            GoCaxon_z,
            r_glom,
            n_conn_goc,
            layer_thickness,
            oob,
        )
        self.scaffold.connect_cells(self, result)
