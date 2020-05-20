import numpy as np
from ..strategy import ConnectionStrategy
from ...exceptions import *
from ...reporting import warn


class ConnectomeGranuleGolgi(ConnectionStrategy):
    """
        Legacy implementation for the connections between Golgi cells and glomeruli.
    """

    casts = {"aa_convergence": int, "pf_convergence": int}

    required = ["aa_convergence", "pf_convergence", "tag_aa", "tag_pf"]

    defaults = {"tag_aa": "ascending_axon_to_golgi", "tag_pf": "parallel_fiber_to_golgi"}

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        golgi_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_granule = int(granules[0, 0])
        r_goc_vol = golgi_cell_type.morphology.dendrite_radius
        oob = (
            self.scaffold.configuration.X * 1000.0
        )  # Any arbitrarily large value outside of simulation volume
        n_connAA = self.aa_convergence
        n_conn_pf = self.pf_convergence
        tot_conn = n_connAA + n_conn_pf
        pf_heights = self.scaffold.appends["cells/ascending_axon_lengths"]

        def connectome_grc_goc(
            first_granule,
            granules,
            golgicells,
            r_goc_vol,
            OoB_value,
            n_connAA,
            n_conn_pf,
            tot_conn,
            scaffold,
        ):
            aa_goc = np.empty((0, 2))
            pf_goc = np.empty((0, 2))
            densityWarningSent = False
            new_granules = np.copy(granules)
            granules_x = new_granules[:, 2]
            granules_z = new_granules[:, 4]
            new_golgicells = np.random.permutation(golgicells)
            if new_granules.shape[0] <= new_golgicells.shape[0]:
                raise ConnectivityError(
                    "The number of granule cells was less than the number of golgi cells. Simulation cannot continue."
                )
            for golgi_id, _, golgi_x, golgi_y, golgi_z in new_golgicells:
                # Distance of this golgi cell to all ascending axons
                distance_vector = ((granules_x - golgi_x) ** 2) + (
                    (granules_z - golgi_z) ** 2
                )
                AA_candidates = np.where((distance_vector).__le__(r_goc_vol ** 2))[
                    0
                ]  # finds indexes of ascending axons that can potentially be connected
                chosen_rand = np.random.permutation(AA_candidates)
                selected_granules = new_granules[chosen_rand]
                selected_distances = np.sqrt(distance_vector[chosen_rand])
                prob = selected_distances / r_goc_vol
                distance_sort = prob.argsort()
                selected_granules = selected_granules[distance_sort]
                prob = prob[distance_sort]
                rolls = np.random.uniform(size=len(selected_granules))
                connectedAA = np.empty(n_connAA)
                idx = 0
                for ind, j in enumerate(selected_granules):
                    if idx < n_connAA:
                        if rolls[ind] > prob[ind]:
                            connectedAA[idx] = j[0]
                            idx += 1
                connectedAA = connectedAA[0:idx]
                good_grc = np.delete(
                    granules, np.array(connectedAA - first_granule, dtype=int), 0
                )
                intersections = (good_grc[:, 2]).__ge__(golgi_x - r_goc_vol) & (
                    good_grc[:, 2]
                ).__le__(golgi_x + r_goc_vol)
                good_pf = np.where(intersections == True)[
                    0
                ]  # finds indexes of granules that can potentially be connected
                # The remaining amount of parallel fibres to connect after subtracting the amount of already connected ascending axons.
                AA_connected_count = len(connectedAA)
                parallelFibersToConnect = tot_conn - AA_connected_count
                # Randomly select parallel fibers to be connected with a GoC, to a maximum of tot_conn connections
                if good_pf.shape[0] < parallelFibersToConnect:
                    connected_pf = np.random.choice(
                        good_pf,
                        min(tot_conn - AA_connected_count, good_pf.shape[0]),
                        replace=False,
                    )
                    totalConnectionsMade = connected_pf.shape[0] + AA_connected_count
                    # Warn the user once if not enough granule cells are present to connect to the Golgi cell.
                    if not densityWarningSent:
                        densityWarningSent = True
                        warn(
                            "The granule cell density is too low compared to the Golgi cell density to make physiological connections!",
                            ConnectivityWarning,
                        )
                else:
                    connected_pf = np.random.choice(
                        good_pf, tot_conn - len(connectedAA), replace=False
                    )
                    totalConnectionsMade = tot_conn
                PF_connected_count = connected_pf.shape[0]
                pf_idx = good_grc[connected_pf, :]
                matrix_aa = np.zeros((AA_connected_count, 2))
                matrix_pf = np.zeros((PF_connected_count, 2))
                matrix_pf[0:PF_connected_count, 0] = pf_idx[:, 0]
                matrix_aa[0:AA_connected_count, 0] = connectedAA
                matrix_pf[:, 1] = golgi_id
                matrix_aa[:, 1] = golgi_id
                pf_goc = np.vstack((pf_goc, matrix_pf))
                aa_goc = np.vstack((aa_goc, matrix_aa))
                new_granules[((connectedAA.astype(int)) - first_granule), :] = OoB_value
                # End of Golgi cell loop
            aa_goc = aa_goc[aa_goc[:, 1].argsort()]
            pf_goc = pf_goc[
                pf_goc[:, 1].argsort()
            ]  # sorting of the resulting vector on the post-synaptic neurons
            return aa_goc, pf_goc

        result_aa, result_pf = connectome_grc_goc(
            first_granule,
            granules,
            golgis,
            r_goc_vol,
            oob,
            n_connAA,
            n_conn_pf,
            tot_conn,
            self.scaffold,
        )
        self.scaffold.connect_cells(self, result_aa, self.tag_aa)
        self.scaffold.connect_cells(self, result_pf, self.tag_pf)
