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
        compartments = np.empty((0, 2))

        def connectome_goc_grc(golgis, glom_grc, goc_glom):
            # Connect all golgi cells to the granule cells that they share a glomerulus with.
            glom_grc_per_glom = {}
            goc_grc = np.empty((0, 2))
            golgi_ids = golgis[:, 0]
            for golgi_id in golgis[:, 0]:
                # Fetch all the glomeruli this golgi is connected to
                connected_glomeruli = goc_glom[goc_glom[:, 0] == golgi_id, 1]
                # Find all the glomerulus-granule connections involving the glomeruli this golgi is connected to.
                intermediary_indices = [
                    i for i, row in enumerate(glom_grc) if row[0] in connected_glomeruli
                ]
                intermediary_connections = glom_grc[intermediary_indices]
                # Extract the ids of the granule cells from these connections
                target_granules = [row[1] for row in intermediary_connections]
                goc_grc = np.vstack(
                    (
                        goc_grc,
                        # Create a matrix with 2 columns where the 1st column is the golgi id
                        np.column_stack(
                            (
                                golgi_id * np.ones(len(target_granules)),
                                # and the 2nd column is all granules connected to one of the glomeruli the golgi is connected to.
                                target_granules,
                            )
                        ),
                    )
                )
                if self.detailed:
                    nonlocal compartments
                    compartments = np.vstack(
                        (
                            compartments,
                            np.column_stack(
                                (
                                    self.axon[
                                        np.random.randint(
                                            0, len(self.axon), len(target_granules)
                                        )
                                    ],
                                    glom_grc_compartments[intermediary_indices, 1] - 1,
                                )
                            ),
                        )
                    )

            return goc_grc

        result = connectome_goc_grc(golgis, glom_grc, goc_glom)
        if self.detailed:
            morpho_map = [
                self.from_cell_types[0].list_all_morphologies()[0],
                self.to_cell_types[0].list_all_morphologies()[0],
            ]
            morphologies = np.zeros((len(result), 2))
            morphologies[:, 1] += 1
            self.scaffold.connect_cells(
                self,
                result,
                morphologies=morphologies,
                compartments=compartments,
                morpho_map=morpho_map,
            )
        else:
            self.scaffold.connect_cells(self, result)
