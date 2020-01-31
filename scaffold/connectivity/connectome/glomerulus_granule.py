import numpy as np
from ..strategy import TouchingConvergenceDivergence


class ConnectomeGlomerulusGranule(TouchingConvergenceDivergence):
    """
        Legacy implementation for the connections between glomeruli and granule cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        from_cell_type = self.from_cell_types[0]
        to_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[from_cell_type.name]
        granules = self.scaffold.cells_by_type[to_cell_type.name]
        dend_len = to_cell_type.morphology.dendrite_length
        n_conn_glom = self.convergence
        first_glomerulus = int(glomeruli[0, 0])

        def connectome_glom_grc(
            first_glomerulus, glomeruli, granules, dend_len, n_conn_glom
        ):
            """
                Legacy code block to connect glomeruli to granule cells
            """
            glom_x = glomeruli[:, 2]
            glom_y = glomeruli[:, 3]
            glom_z = glomeruli[:, 4]
            results = np.empty((granules.shape[0] * n_conn_glom, 2))
            next_index = 0
            # Find glomeruli to connect to each granule cell
            for gran_id, gran_type, gran_x, gran_y, gran_z in granules:
                # Use a naive approach to find all glomeruli at a maximum distance of `dendrite_length`
                distance_vector = (
                    ((glom_x - gran_x) ** 2)
                    + ((glom_y - gran_y) ** 2)
                    + ((glom_z - gran_z) ** 2)
                    - (dend_len ** 2)
                )
                good_gloms = np.where((distance_vector < 0.0) == True)[
                    0
                ]  # indexes of glomeruli that can potentially be connected
                good_gloms_len = len(good_gloms)
                # Do we find more than enough candidates?
                if good_gloms_len > n_conn_glom:  # Yes: select the closest ones
                    # Get the distances of the glomeruli within range
                    gloms_distance = distance_vector[good_gloms]
                    # Sort the good glomerulus id vector by the good glomerulus distance vector
                    connected_gloms = good_gloms[gloms_distance.argsort()]
                    connected_glom_len = n_conn_glom
                else:  # No: select all of them
                    connected_gloms = good_gloms
                    connected_glom_len = good_gloms_len
                # Connect the selected glomeruli to the current gran_id
                for i in range(connected_glom_len):
                    # Add the first_glomerulus id to convert their local id to their real simulation id
                    results[next_index + i] = [
                        connected_gloms[i] + first_glomerulus,
                        gran_id,
                    ]
                # Move up the internal array pointer
                next_index += connected_glom_len
            # Truncate the pre-allocated array to the internal array pointer.
            return results[0:next_index, :]

        # Execute legacy code and add the connection matrix it returns to the scaffold.
        connectome = connectome_glom_grc(
            first_glomerulus, glomeruli, granules, dend_len, n_conn_glom
        )
        self.scaffold.connect_cells(self, connectome)
