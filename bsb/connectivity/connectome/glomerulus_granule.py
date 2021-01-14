import numpy as np, random
from ..strategy import ConnectionStrategy
from ...exceptions import ConfigurationError, ConnectivityError


class ConnectomeGlomerulusGranule(ConnectionStrategy):
    """
    Legacy implementation for the connections between glomeruli and granule cells.
    """

    casts = {"detailed": bool, "convergence": int}
    defaults = {"detailed": False}
    required = ["convergence"]

    def validate(self):
        if self.detailed:
            morphologies = self.to_cell_types[0].list_all_morphologies()
            if not morphologies:
                raise ConfigurationError(
                    "Can't create detailed glomerulus to granule connections without any morphologies for the granule cell."
                )
            elif len(morphologies) > 1:
                raise NotImplementedError(
                    "Detailed glomerulus to granule connections can only be made for a single morphology."
                    + " (Requires the selection of morphologies to be moved from the connection module to the placement module)"
                )
            mr = self.scaffold.morphology_repository
            morphology = mr.get_morphology(morphologies[0])
            dendritic_compartments = morphology.get_compartments(["dendrites"])
            dendrites = {}
            for c in dendritic_compartments:
                # Store the last found compartment of each dendrite
                dendrites[c.section_id] = c
            self.dendritic_claws = [c.id for c in dendrites.values()]
            self.morphology = morphology

    def connect(self):
        # Gather information for the legacy code block below.
        from_cell_type = self.from_cell_types[0]
        to_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[from_cell_type.name]
        granules = self.scaffold.cells_by_type[to_cell_type.name]
        dend_len = to_cell_type.morphology.dendrite_length
        n_conn_glom = self.convergence
        first_glomerulus = int(glomeruli[0, 0])
        mf_to_glom = self.scaffold.cell_connections_by_tag["mossy_to_glomerulus"]
        glom_mf_map = {v: k for k, v in mf_to_glom}

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
                # Indices of glomeruli that can potentially be connected
                good_gloms = np.where((distance_vector < 0.0) == True)[0]
                had_mf = set()
                candidates = []
                for g in np.random.permutation(good_gloms):
                    mf = glom_mf_map[g + first_glomerulus]
                    if mf in had_mf:
                        continue
                    had_mf.add(mf)
                    candidates.append(g)
                good_gloms = np.array(candidates)
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
            return results[:next_index, :]

        # Execute legacy code and add the connection matrix it returns to the scaffold.
        connectome = connectome_glom_grc(
            first_glomerulus, glomeruli, granules, dend_len, n_conn_glom
        )
        if self.detailed:
            # Add morphology & compartment information
            morpho_map = [self.morphology.morphology_name]
            morphologies = np.zeros((len(connectome), 2))
            # Store a map between the granule cell ids and the available claw compartment ids
            granule_dendrite_occupation = {
                g[0]: self.dendritic_claws.copy() for g in granules
            }
            # Shuffle the order in which the dendrites will be selected by glomeruli
            for l in granule_dendrite_occupation.values():
                random.shuffle(l)
            compartments = []
            from time import time

            t = time()
            for i in range(len(connectome)):
                granule_id = connectome[i, 1]
                try:
                    unoccupied_claw = granule_dendrite_occupation[granule_id].pop()
                except IndexError:
                    raise ConnectivityError(
                        "Attempt to connect a glomerulus to a fully saturated granule cell."
                    )
                compartments.append([0, unoccupied_claw])
            self.scaffold.connect_cells(
                self,
                connectome,
                morphologies=morphologies,
                compartments=np.array(compartments),
                morpho_map=morpho_map,
            )
        else:
            self.scaffold.connect_cells(self, connectome)
