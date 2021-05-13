import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeGolgiGranule(ConnectionStrategy):
    """
    Legacy implementation for the connections between Golgi cells and granule cells.
    """

    casts = {"detailed": bool}
    defaults = {"detailed": False}

    def validate(self):
        if self.detailed:
            morphologies = self.from_cell_types[0].list_all_morphologies()
            if not morphologies:
                raise ConfigurationError(
                    "Can't create detailed golgi to granule connections without any morphologies for the golgi cell."
                )
            elif len(morphologies) > 1:
                raise NotImplementedError(
                    "Detailed golgi to granule connections can only be made for a single golgi morphology."
                    + " (Requires the selection of morphologies to be moved from the connection module to the placement module)"
                )
            mr = self.scaffold.morphology_repository
            morphology = mr.get_morphology(morphologies[0])
            axonic_compartments = morphology.get_compartments(["axon"])
            self.axon = np.array([c.id for c in axonic_compartments])
            self.morphology = morphology

    def connect(self):
        # Gather information for the legacy code block below.
        glom_grc = self.scaffold.cell_connections_by_tag["glomerulus_to_granule"]
        glom_ids = self.scaffold.get_placement_set("glomerulus").identifiers
        goc_glom = self.scaffold.cell_connections_by_tag["golgi_to_glomerulus"]
        golgi_type = self.from_cell_types[0]
        golgis = self.scaffold.cells_by_type[golgi_type.name]
        if self.detailed:
            # Fetch the connection compartments of glomerulus to granule connections.
            try:
                glom_grc_compartments = self.scaffold.connection_compartments[
                    "glomerulus_to_granule"
                ]
            except KeyError:
                raise RuntimeError(
                    "Missing the glomerulus to granule connection compartments."
                )
        # Index the glom to granule dense matrix to an adjacency list
        glom_target_map = {gid: [] for gid in glom_ids}
        for glom, grc in glom_grc:
            glom_target_map[glom].append(grc)

        if self.detailed:
            # Lookup the grc compartments associated with each glom-grc conn
            glom_comp_map = {gid: [] for gid in glom_ids}
            for i, glom in enumerate(glom_grc[:, 0]):
                glom_comp_map[glom].append(glom_grc_compartments[i, 1])

        # Lookup the grc targets of each golgi-glom in the order they appear.
        grc_via_glom = np.vectorize(glom_target_map.get, otypes=[np.ndarray])(
            goc_glom[:, 1]
        )

        if self.detailed:
            comp_via_glom = np.vectorize(glom_comp_map.get, otypes=[np.ndarray])(
                goc_glom[:, 1]
            )

        # Sum the targets to preallocate total size
        malloc = sum(map(len, grc_via_glom))
        connections = np.empty((malloc, 2))
        ptr = 0
        # Iterate over the golgis and their grc targets (as lookup up in order
        # via the gloms) and fill them into the connections array
        for goc, grcs in zip(goc_glom[:, 0], grc_via_glom):
            s = len(grcs)
            if not s:
                # Skip any weird slicing of empty target sets
                continue
            connections[ptr : (ptr + s), 0] = goc
            connections[ptr : (ptr + s), 1] = grcs
            ptr += s

        if self.detailed:
            compartments = np.empty((malloc, 2))
            # Assign random axonal segments
            compartments[:, 0] = np.random.choice(self.axon, malloc)
            ptr = 0
            # Assign the glom associated dendrites
            for comps in comp_via_glom:
                s = len(comps)
                if not s:
                    # Skip any weird slicing of empty target sets
                    continue
                compartments[ptr : (ptr + s), 1] = comps
                ptr += s
            # Make a bad map ignoring all but the first (only) morphologies
            # of each type. Sorry future me <3
            morpho_map = [
                self.from_cell_types[0].list_all_morphologies()[0],
                self.to_cell_types[0].list_all_morphologies()[0],
            ]
            morphologies = np.zeros((len(compartments), 2))
            morphologies[:, 1] = 1
            self.scaffold.connect_cells(
                self,
                connections,
                morphologies=morphologies,
                compartments=compartments,
                morpho_map=morpho_map,
            )
        else:
            self.scaffold.connect_cells(self, connections)
