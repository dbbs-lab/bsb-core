import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeGapJunctions(ConnectionStrategy):
    """
        Legacy implementation for gap junctions between a cell type.
    """

    casts = {"limit_xy": float, "limit_z": float, "divergence": int}

    required = ["limit_xy", "limit_z", "divergence"]

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        from_cell_type = self.from_cell_types[0]
        from_cells = self.scaffold.cells_by_type[from_cell_type.name]
        first_cell = int(from_cells[0, 0])
        limit_xy = self.limit_xy
        limit_z = self.limit_z
        divergence = self.divergence

        def gap_junctions(cells, d_xy, d_z, dc_gj):
            gj_sc = np.zeros((cells.shape[0] * dc_gj, 2))
            gj_i = 0
            cells_x = cells[:, 2]
            cells_y = cells[:, 3]
            cells_z = cells[:, 4]

            for (
                id,
                type,
                x,
                y,
                z,
            ) in (
                cells
            ):  # for each stellate cell calculate the distance with every other cell of the same type in the volume, then choose 4 of them

                idx = 1

                # find all cells that satisfy the distance condition
                constraint_vector = (
                    (np.absolute(cells_z - z)).__lt__(d_z)
                    & (np.absolute(cells_z - z)).__ne__(0)
                    & (np.sqrt((cells_x - x) ** 2 + (cells_y - y) ** 2)).__lt__(d_xy)
                )
                good_sc = np.where(constraint_vector)[
                    0
                ]  # indexes of stellate cells that can potentially be connected
                chosen_rand = np.random.permutation(good_sc)
                candidates = cells[chosen_rand]

                for j in candidates:

                    if idx <= dc_gj:

                        ra = np.random.random()
                        if (ra).__gt__((np.absolute(j[4] - z)) / float(d_z)) & (
                            ra
                        ).__gt__(
                            (np.sqrt((j[2] - x) ** 2 + (j[3] - y) ** 2)) / float(d_xy)
                        ):
                            idx += 1
                            gj_sc[gj_i, 0] = id
                            gj_sc[gj_i, 1] = j[0]
                            gj_i += 1

            return gj_sc[0:gj_i]

        result = gap_junctions(from_cells, limit_xy, limit_z, divergence)
        self.scaffold.connect_cells(self, result)


class ConnectomeGapJunctionsGolgi(ConnectionStrategy):
    """
        Legacy implementation for Golgi cell gap junctions.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        golgi_cell_type = self.from_cell_types[0]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_golgi = int(golgis[0, 0])
        r_goc_vol = golgi_cell_type.morphology.dendrite_radius
        GoCaxon_x = golgi_cell_type.morphology.axon_x
        GoCaxon_y = golgi_cell_type.morphology.axon_y
        GoCaxon_z = golgi_cell_type.morphology.axon_z

        def connectome_gj_goc(
            r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgicells, first_golgi
        ):
            gj_goc = np.zeros((golgis.shape[0] ** 2, 2))
            start_index = 0
            golgi_x = golgicells[:, 2]
            golgi_y = golgicells[:, 3]
            golgi_z = golgicells[:, 4]
            self_index = 0
            for (
                i
            ) in (
                golgicells
            ):  # for each Golgi find all cells of the same type that, through their dendritic tree, fall into its axonal tree

                bool_vector = (np.absolute(golgi_x - i[2])).__le__(
                    r_goc_vol + (GoCaxon_x / 2.0)
                )
                bool_vector[self_index] = False  # Don't connect a golgi cell to itself
                bool_vector = bool_vector & (np.absolute(golgi_y - i[3])).__le__(
                    r_goc_vol + (GoCaxon_y / 2.0)
                )
                bool_vector = bool_vector & (np.absolute(golgi_z - i[4])).__le__(
                    r_goc_vol + (GoCaxon_z / 2.0)
                )

                good_goc = np.where(bool_vector)[
                    0
                ]  # finds indexes of Golgi cells that satisfy all conditions

                end_index = start_index + len(good_goc)
                gj_goc[start_index:end_index, 0] = i[0]
                gj_goc[start_index:end_index, 1] = golgicells[good_goc][:, 0]
                start_index = end_index
                self_index += 1

            return gj_goc[0:end_index]

        result = connectome_gj_goc(
            r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgis, first_golgi
        )
        self.scaffold.connect_cells(self, result)
