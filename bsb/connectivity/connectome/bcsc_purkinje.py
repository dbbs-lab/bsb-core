import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeBCSCPurkinje(ConnectionStrategy):
    """
    Legacy implementation for the connections between basket cells,stellate cells and purkinje cells.
    """

    casts = {"limit_x": float, "limit_z": float, "divergence": int, "convergence": int}

    required = ["limit_x", "limit_z", "divergence", "convergence", "tag_bc", "tag_sc"]

    defaults = {"tag_bc": "basket_to_purkinje", "tag_sc": "stellate_to_purkinje"}

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        basket_cell_type = self.from_cell_types[0]
        stellate_cell_type = self.from_cell_types[1]
        purkinje_cell_type = self.to_cell_types[0]
        baskets = self.scaffold.cells_by_type[basket_cell_type.name]
        stellates = self.scaffold.cells_by_type[stellate_cell_type.name]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        first_basket = int(baskets[0, 0])
        first_stellate = int(stellates[0, 0])
        distx = self.limit_x
        distz = self.limit_z
        conv = self.convergence

        def connectome_bc_sc_pc(
            first_basket,
            first_stellate,
            basketcells,
            stellates,
            purkinjes,
            distx,
            distz,
            conv,
        ):
            n_purkinje = len(purkinjes)
            bc_pc = np.empty((n_purkinje * conv, 2))
            sc_pc = np.empty((n_purkinje * conv, 2))
            bc_i = 0
            sc_i = 0

            baskets_x = baskets[:, 2]
            baskets_z = baskets[:, 4]
            stellates_x = stellates[:, 2]
            stellates_z = stellates[:, 4]
            for (
                p_id,
                p_type,
                p_x,
                p_y,
                p_z,
            ) in (
                purkinjes
            ):  # for all Purkinje cells: calculate which basket and stellate cells can be connected, then choose 20 of them for each typology

                idx_bc = 1
                idx_sc = 1

                # find all cells that satisfy the distance condition for both types
                bc_matrix = (np.absolute(baskets_z - p_z)).__lt__(distx) & (
                    np.absolute(baskets_x - p_x)
                ).__lt__(distz)
                sc_matrix = (np.absolute(stellates_z - p_z)).__lt__(distz) & (
                    np.absolute(stellates_x - p_x)
                ).__lt__(distx)

                good_bc = np.where(bc_matrix)[
                    0
                ]  # indexes of basket cells that can potentially be connected
                good_sc = np.where(sc_matrix)[
                    0
                ]  # indexes of stellate cells that can potentially be connected

                chosen_rand_bc = np.random.permutation(good_bc)
                good_bc_matrix = basketcells[chosen_rand_bc]
                chosen_rand_sc = np.random.permutation(good_sc)
                good_sc_matrix = stellates[chosen_rand_sc]

                # basket cells connectivity
                for j in good_bc_matrix:

                    if idx_bc <= conv:

                        ra = np.random.random()
                        if (ra).__gt__((np.absolute(j[4] - p_z)) / distx) & (ra).__gt__(
                            (np.absolute(j[2] - p_x)) / distz
                        ):
                            idx_bc += 1
                            bc_pc[bc_i, 0] = j[0]
                            bc_pc[bc_i, 1] = p_id
                            bc_i += 1

                # stellate cells connectivity
                for k in good_sc_matrix:

                    if idx_sc <= conv:

                        ra = np.random.random()
                        if (ra).__gt__((np.absolute(k[4] - p_z)) / distz) & (ra).__gt__(
                            (np.absolute(k[2] - p_x)) / distx
                        ):
                            idx_sc += 1
                            sc_pc[sc_i, 0] = k[0]
                            sc_pc[sc_i, 1] = p_id
                            sc_i += 1

            return bc_pc[0:bc_i], sc_pc[0:sc_i]

        result_bc, result_sc = connectome_bc_sc_pc(
            first_basket,
            first_stellate,
            baskets,
            stellates,
            purkinjes,
            distx,
            distz,
            conv,
        )
        self.scaffold.connect_cells(
            self,
            result_bc,
            self.tag_bc,
            meta={"from_cell_types": [basket_cell_type.name]},
        )
        self.scaffold.connect_cells(
            self,
            result_sc,
            self.tag_sc,
            meta={"from_cell_types": [stellate_cell_type.name]},
        )
