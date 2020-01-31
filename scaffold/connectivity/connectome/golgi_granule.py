import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeGolgiGranule(ConnectionStrategy):
    """
        Legacy implementation for the connections between Golgi cells and granule cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        glom_grc = self.scaffold.cell_connections_by_tag["glomerulus_to_granule"]
        goc_glom = self.scaffold.cell_connections_by_tag["golgi_to_glomerulus"]
        golgi_type = self.from_cell_types[0]
        golgis = self.scaffold.cells_by_type[golgi_type.name]

        def connectome_goc_grc(golgis, glom_grc, goc_glom):
            # Connect all golgi cells to the granule cells that they share a glomerulus with.
            glom_grc_per_glom = {}
            goc_grc = np.empty((0, 2))
            golgi_ids = golgis[:, 0]
            for golgi_id in golgis[:, 0]:
                # Fetch all the glomeruli this golgi is connected to
                connected_glomeruli = goc_glom[goc_glom[:, 0] == golgi_id, 1]
                # Append a new set of connections after the existing set of goc_grc connections.
                connected_granules_via_gloms = list(
                    map(
                        lambda row: row[1],
                        filter(lambda row: row[0] in connected_glomeruli, glom_grc),
                    )
                )
                goc_grc = np.vstack(
                    (
                        goc_grc,
                        # Create a matrix with 2 columns where the 1st column is the golgi id
                        np.column_stack(
                            (
                                golgi_id * np.ones(len(connected_granules_via_gloms)),
                                # and the 2nd column is all granules connected to one of the glomeruli the golgi is connected to.
                                connected_granules_via_gloms,
                            )
                        ),
                    )
                )

            return goc_grc

        result = connectome_goc_grc(golgis, glom_grc, goc_glom)
        self.scaffold.connect_cells(self, result)
