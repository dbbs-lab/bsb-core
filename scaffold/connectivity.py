import abc
from .helpers import (
    ConfigurableClass,
    DistributionConfiguration,
    assert_attr_in,
    SortableByAfter,
)
from .postprocessing import get_parallel_fiber_heights, get_dcn_rotations
from .models import ConnectivitySet
import numpy as np
from random import choice as random_element, sample as sample_elements
from .exceptions import MissingMorphologyException, ConnectivityWarning
from sklearn.cluster import KMeans


class SimulationPlaceholder:
    pass


class ConnectionStrategy(ConfigurableClass, SortableByAfter):
    def __init__(self):
        super().__init__()
        self.simulation = SimulationPlaceholder()
        self.tags = []

    @abc.abstractmethod
    def connect(self):
        pass

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of connection types required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def has_after(self):
        return hasattr(self, "after")

    def create_after(self):
        self.after = []

    def get_connection_matrices(self):
        return [self.scaffold.cell_connections_by_tag[tag] for tag in self.tags]

    def get_connectivity_sets(self):
        return [ConnectivitySet(self.scaffold.output_formatter, tag) for tag in self.tags]


class ReciprocalGolgiGlomerulus(ConnectionStrategy):
    def validate(self):
        pass

    def connect(self):
        pass


class TouchingConvergenceDivergence(ConnectionStrategy):
    casts = {"divergence": int, "convergence": int}

    required = ["divergence", "convergence"]

    def validate(self):
        pass

    def connect(self):
        pass


class TouchConnect(ConnectionStrategy):
    def validate(self):
        pass

    def connect(self):
        pass


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


class ConnectomeGlomerulusGolgi(TouchingConvergenceDivergence):
    """
        Legacy implementation for the connections between Golgi cells and glomeruli.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        glomerulus_cell_type = self.from_cell_types[0]
        golgi_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_glomerulus = int(glomeruli[0, 0])
        r_goc_vol = golgi_cell_type.morphology.dendrite_radius

        def connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol):
            glom_bd = np.zeros((0, 2))
            glom_x = glomeruli[:, 2]
            glom_y = glomeruli[:, 3]
            glom_z = glomeruli[:, 4]

            # for all Golgi cells: calculate which glomeruli fall into the volume of GoC basolateral dendrites, then choose 40 of them for the connection and delete them from successive computations, since 1 axon is connected to 1 GoC
            for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in golgicells:
                # Geometric constraints: glom less than `r_goc_vol` away from golgi and golgi cell soma above glom.
                volume_matrix = (
                    ((glom_x - golgi_x) ** 2)
                    + ((glom_y - golgi_y) ** 2)
                    + ((glom_z - golgi_z) ** 2)
                    - (r_goc_vol ** 2)
                ).__le__(0) & glom_y.__le__(golgi_y)
                good_gloms = np.where(volume_matrix == True)[
                    0
                ]  # finds indexes of granules that can potentially be connected
                connected_gloms = (
                    good_gloms + first_glomerulus
                )  # Translate local id to simulation id

                matrix = np.zeros((len(good_gloms), 2))
                matrix[:, 0] = connected_gloms  # from cell
                matrix[:, 1] = golgi_id  # to cell
                glom_bd = np.vstack((glom_bd, matrix))

            return glom_bd

        connectome = connectome_glom_goc(first_glomerulus, glomeruli, golgis, r_goc_vol)
        self.scaffold.connect_cells(self, connectome)


class ConnectomeGolgiGlomerulus(TouchingConvergenceDivergence):
    """
        Legacy implementation for the connections between glomeruli and Golgi cells.
    """

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

            # for all Golgi cells: calculate which glomeruli fall into the area of GoC axon, then choose 40 of them for the connection and delete them from successive computations, since 1 glomerulus must be connected to only 1 GoC
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
        pf_heights = get_parallel_fiber_heights(
            self.scaffold, granule_cell_type.morphology, granules
        )
        self.scaffold.append_dset("cells/ascending_axon_lengths", data=pf_heights)

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
                raise Exception(
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
                        if scaffold.configuration.verbosity > 0:
                            scaffold.warn(
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


class ConnectomeAscAxonPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connections between ascending axons and purkinje cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        purkinje_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        first_granule = int(granules[0, 0])
        OoB_value = (
            self.scaffold.configuration.X * 1000.0
        )  # Any arbitrarily large value outside of simulation volume
        purkinje_extension_x = purkinje_cell_type.placement.extension_x
        purkinje_extension_z = purkinje_cell_type.placement.extension_z

        def connectome_aa_pc(first_granule, granules, purkinjes, x_pc, z_pc, OoB_value):
            aa_pc = np.zeros((0, 2))
            new_granules = np.copy(granules)

            # for all Purkinje cells: calculate and choose which granules fall into the area of PC dendritic tree, then delete them from successive computations, since 1 ascending axon is connected to only 1 PC
            for i in purkinjes:
                # ascending axon falls into the z range of values?
                bool_vector = (new_granules[:, 4]).__ge__(i[4] - z_pc / 2.0) & (
                    new_granules[:, 4]
                ).__le__(i[4] + z_pc / 2.0)
                # ascending axon falls into the x range of values?
                bool_vector = bool_vector & (
                    (new_granules[:, 2]).__ge__(i[2] - x_pc / 2.0)
                    & (new_granules[:, 2]).__le__(i[2] + x_pc / 2.0)
                )
                good_aa = np.where(bool_vector)[
                    0
                ]  # finds indexes of ascending axons that, on the selected axis, have the correct sum value

                # construction of the output matrix: the first column has the GrC id, while the second column has the PC id
                matrix = np.zeros((len(good_aa), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_aa + first_granule
                aa_pc = np.vstack((aa_pc, matrix))

                new_granules[
                    good_aa, :
                ] = OoB_value  # update the granules matrix used for computation by deleting the coordinates of connected ones

            return aa_pc

        result = connectome_aa_pc(
            first_granule,
            granules,
            purkinjes,
            purkinje_extension_x,
            purkinje_extension_z,
            OoB_value,
        )
        self.scaffold.connect_cells(self, result)


class ConnectomePFPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connections between parallel fibers and purkinje cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        purkinje_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        first_granule = int(granules[0, 0])
        purkinje_extension_x = purkinje_cell_type.placement.extension_x

        def connectome_pf_pc(first_granule, granules, purkinjes, x_pc):
            pf_pc = np.zeros((0, 2))
            # for all Purkinje cells: calculate and choose which parallel fibers fall into the area of PC dendritic tree (then delete them from successive computations, since 1 parallel fiber is connected to a maximum of PCs)
            for i in purkinjes:
                # which parallel fibers fall into the x range of values?
                bool_matrix = (granules[:, 2]).__ge__(i[2] - x_pc / 2.0) & (
                    granules[:, 2]
                ).__le__(
                    i[2] + x_pc / 2.0
                )  # CAMBIARE IN new_granules SE VINCOLO SU 30 pfs
                good_pf = np.where(bool_matrix)[
                    0
                ]  # finds indexes of parallel fibers that, on the selected axis, satisfy the condition

                # construction of the output matrix: the first column has the GrC id, while the second column has the PC id
                matrix = np.zeros((len(good_pf), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_pf + first_granule
                pf_pc = np.vstack((pf_pc, matrix))

            return pf_pc

        result = connectome_pf_pc(
            first_granule, granules, purkinjes, purkinje_extension_x
        )
        self.scaffold.connect_cells(self, result)


class ConnectomePFInterneuron(ConnectionStrategy):
    """
        Legacy implementation for the connections between parallel fibers and a molecular layer interneuron cell_type.
    """

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        granule_cell_type = self.from_cell_types[0]
        interneuron_cell_type = self.to_cell_types[0]
        granules = self.scaffold.cells_by_type[granule_cell_type.name]
        interneurons = self.scaffold.cells_by_type[interneuron_cell_type.name]
        first_granule = int(granules[0, 0])
        dendrite_radius = interneuron_cell_type.morphology.dendrite_radius
        pf_heights = (
            self.scaffold.appends["cells/ascending_axon_lengths"][:, 1] + granules[:, 3]
        )  # Add granule Y to height of its pf

        def connectome_pf_inter(first_granule, interneurons, granules, r_sb, h_pf):
            pf_interneuron = np.zeros((0, 2))

            for (
                i
            ) in (
                interneurons
            ):  # for each interneuron find all the parallel fibers that fall into the sphere with centre the cell soma and appropriate radius

                # find all cells that satisfy the condition
                interneuron_matrix = (
                    ((granules[:, 2] - i[2]) ** 2) + ((h_pf - i[3]) ** 2) - (r_sb ** 2)
                ).__le__(0)
                good_pf = np.where(interneuron_matrix)[
                    0
                ]  # indexes of interneurons that can potentially be connected

                matrix = np.zeros((len(good_pf), 2))
                matrix[:, 1] = i[0]
                matrix[:, 0] = good_pf + first_granule
                pf_interneuron = np.vstack((pf_interneuron, matrix))

            return pf_interneuron

        result = connectome_pf_inter(
            first_granule, interneurons, granules, dendrite_radius, pf_heights
        )
        self.scaffold.connect_cells(self, result)


class ConnectomeBCSCPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connections between stellate cells, basket cells and purkinje cells.
    """

    casts = {"limit_x": float, "limit_z": float, "divergence": int, "convergence": int}

    required = ["limit_x", "limit_z", "divergence", "convergence", "tag_sc", "tag_bc"]

    defaults = {"tag_sc": "stellate_to_purkinje", "tag_bc": "basket_to_purkinje"}

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        basket_cell_type = self.from_cell_types[0]
        stellate_cell_type = self.from_cell_types[1]
        purkinje_cell_type = self.to_cell_types[0]
        stellates = self.scaffold.cells_by_type[stellate_cell_type.name]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        baskets = self.scaffold.cells_by_type[basket_cell_type.name]
        first_stellate = int(stellates[0, 0])
        first_basket = int(baskets[0, 0])
        distx = self.limit_x
        distz = self.limit_z
        conv = self.convergence

        def connectome_sc_bc_pc(
            first_stellate,
            first_basket,
            basketcells,
            stellates,
            purkinjes,
            distx,
            distz,
            conv,
        ):
            n_purkinje = len(purkinjes)
            sc_pc = np.empty((n_purkinje * conv, 2))
            bc_pc = np.empty((n_purkinje * conv, 2))
            bc_i = 0
            sc_i = 0

            stellates_x = stellates[:, 2]
            stellates_z = stellates[:, 4]
            baskets_x = baskets[:, 2]
            baskets_z = baskets[:, 4]
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
                sc_matrix = (np.absolute(stellates_z - p_z)).__lt__(distz) & (
                    np.absolute(stellates_x - p_x)
                ).__lt__(distx)
                bc_matrix = (np.absolute(baskets_z - p_z)).__lt__(distx) & (
                    np.absolute(baskets_x - p_x)
                ).__lt__(distz)

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

            return sc_pc[0:sc_i], bc_pc[0:bc_i]

        result_sc, result_bc = connectome_sc_bc_pc(
            first_stellate,
            first_basket,
            baskets,
            stellates,
            purkinjes,
            distx,
            distz,
            conv,
        )
        self.scaffold.connect_cells(
            self,
            result_sc,
            self.tag_sc,
            meta={"from_cell_types": [stellate_cell_type.name]},
        )
        self.scaffold.connect_cells(
            self,
            result_bc,
            self.tag_bc,
            meta={"from_cell_types": [basket_cell_type.name]},
        )


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


class ConnectomePurkinjeDCN(ConnectionStrategy):
    """
        Legacy implementation for the connection between purkinje cells and DCN cells.
        Also rotates the dendritic trees of the DCN.
    """

    casts = {"divergence": int}

    required = ["divergence"]

    def validate(self):
        pass

    def connect(self):
        # Gather information for the legacy code block below.
        purkinje_cell_type = self.from_cell_types[0]
        dcn_cell_type = self.to_cell_types[0]
        purkinjes = self.scaffold.cells_by_type[purkinje_cell_type.name]
        dcn_cells = self.scaffold.cells_by_type[dcn_cell_type.name]
        dcn_angles = get_dcn_rotations(dcn_cells)
        self.scaffold.append_dset("cells/dcn_orientations", data=dcn_angles)
        if len(dcn_cells) == 0:
            return
        first_dcn = int(dcn_cells[0, 0])
        divergence = self.divergence

        def connectome_pc_dcn(first_dcn, purkinjes, dcn_cells, div_pc, dend_tree_coeff):
            pc_dcn = np.zeros((0, 2))

            for (
                i
            ) in (
                purkinjes
            ):  # for all Purkinje cells: calculate the distance with the area around glutamatergic DCN cells soma, then choose 4-5 of them

                distance = np.zeros((dcn_cells.shape[0]))
                distance = (
                    np.absolute(
                        (dend_tree_coeff[:, 0] * i[2])
                        + (dend_tree_coeff[:, 1] * i[3])
                        + (dend_tree_coeff[:, 2] * i[4])
                        + dend_tree_coeff[:, 3]
                    )
                ) / (
                    np.sqrt(
                        (dend_tree_coeff[:, 0] ** 2)
                        + (dend_tree_coeff[:, 1] ** 2)
                        + (dend_tree_coeff[:, 2] ** 2)
                    )
                )

                dist_matrix = np.zeros((dcn_cells.shape[0], 2))
                dist_matrix[:, 1] = dcn_cells[:, 0]
                dist_matrix[:, 0] = distance
                dcn_dist = np.random.permutation(dist_matrix)

                # If the number of DCN cells are less than the divergence value, all neurons are connected to the corresponding PC
                if dcn_cells.shape[0] < div_pc:
                    matrix = np.zeros((dcn_cells.shape[0], 2))
                    matrix[:, 0] = i[0]
                    matrix[:, 1] = dcn_cells[:, 0]
                    pc_dcn = np.vstack((pc_dcn, matrix))

                else:
                    if np.random.rand() > 0.5:

                        connected_f = dcn_dist[0:div_pc, 1]
                        connected_dist = dcn_dist[0:div_pc, 0]
                        connected_provv = connected_f.astype(int)
                        connected_dcn = connected_provv

                        # construction of the output matrix: the first column has the  PC index, while the second column has the connected DCN cell index
                        matrix = np.zeros((div_pc, 2))
                        matrix[:, 0] = i[0]
                        matrix[:, 1] = connected_dcn
                        pc_dcn = np.vstack((pc_dcn, matrix))

                    else:
                        connected_f = dcn_dist[0 : (div_pc - 1), 1]
                        connected_dist = dcn_dist[0 : (div_pc - 1), 0]
                        connected_provv = connected_f.astype(int)
                        connected_dcn = connected_provv

                        matrix = np.zeros(((div_pc - 1), 2))
                        matrix[:, 0] = i[0]
                        matrix[:, 1] = connected_dcn
                        pc_dcn = np.vstack((pc_dcn, matrix))
            return pc_dcn

        results = connectome_pc_dcn(
            first_dcn, purkinjes, dcn_cells, divergence, dcn_angles
        )
        self.scaffold.connect_cells(self, results)


class ConnectomeMossyDCN(TouchingConvergenceDivergence):
    """
        Implementation for the connection between mossy fibers and DCN cells.
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        mossy_cell_type = self.from_cell_types[0]
        dcn_cell_type = self.to_cell_types[0]
        mossy = self.scaffold.entities_by_type[mossy_cell_type.name]
        dcn_cells = self.scaffold.cells_by_type[dcn_cell_type.name]

        convergence = self.convergence
        if convergence > len(mossy):
            convergence = len(mossy)
            self.scaffold.warn(
                "Convergence for MF-DCN saturated at MF number. Network too small",
                ConnectivityWarning,
            )

        mf_dcn = np.zeros((convergence * len(dcn_cells), 2))
        for i, dcn in enumerate(dcn_cells):
            connected_mfs = np.random.choice(mossy, convergence, replace=False)
            range_i = range(i * convergence, (i + 1) * convergence)
            mf_dcn[range_i, 0] = connected_mfs.astype(int)
            mf_dcn[range_i, 1] = dcn[0]

        self.scaffold.connect_cells(self, mf_dcn)


class ConnectomeGeneralConvergence(TouchingConvergenceDivergence):
    """
        Implementation of a general convergence connectivity between
        two populations of cells (this does not work with entities)
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        pre_type = self.from_cell_types[0]
        post_type = self.to_cell_types[0]
        pre = self.scaffold.cells_by_type[pre_type.name]
        post = self.scaffold.cells_by_type[post_type.name]
        convergence = self.convergence

        pre_post = np.zeros((convergence * len(post), 2))
        for i, neuron in enumerate(post):
            connected_pre = np.random.choice(pre[:, 0], convergence, replace=False)
            range_i = range(i * convergence, (i + 1) * convergence)
            pre_post[range_i, 0] = connected_pre.astype(int)
            pre_post[range_i, 1] = neuron[0]

        self.scaffold.connect_cells(self, pre_post)


class ConnectomeIOPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connection between inferior olive and Purkinje cells.
        Purkinje cells are clustered (number of clusters is the number of IO cells), and each clusters
        is innervated by 1 IO cell
    """

    required = ["divergence"]

    def validate(self):
        pass

    def connect(self):
        io_cell_type = self.from_cell_types[0]
        purkinje_cell_type = self.to_cell_types[0]
        io_cells = self.scaffold.cells_by_type[io_cell_type.name]
        purkinje_cells = self.scaffold.cells_by_type[purkinje_cell_type.name]
        convergence = 1  # Purkinje cells should be always constrained to receive signal from only 1 Inferior Olive neuron
        divergence = self.divergence
        tolerance = self.tolerance_divergence

        number_clusters = len(io_cells)
        if number_clusters == 0 or len(purkinje_cells) == 0:
            return
        kmeans = KMeans(n_clusters=number_clusters).fit(purkinje_cells[:, 2:4])
        label_clusters = kmeans.labels_
        target_clusters = {
            i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)
        }
        io_purkinje = np.empty([len(purkinje_cells), 2])
        mi = 0
        for io in range(len(io_cells)):
            target_purkinje_ids = purkinje_cells[target_clusters[io], 0]
            io_ids = np.repeat(io, len(target_clusters[io]))
            nmi = mi + len(target_purkinje_ids)
            io_purkinje[mi:nmi] = np.column_stack((io_ids, target_purkinje_ids))
            mi = nmi

        results = io_purkinje
        self.scaffold.connect_cells(self, results)


class ConnectomeIOMolecular(ConnectionStrategy):
    """
        Legacy implementation for the connection between inferior olive and Molecular layer interneurons.
        As this is a spillover-mediated non-synaptic connection depending on the IO to Purkinje cells, each interneuron connected
        to a PC which is receving input from one IO, is also receiving input from that IO
    """

    def validate(self):
        pass

    def connect(self):
        # Gather connection information
        io_cell_type = self.from_cell_types[0]
        molecular_cell_type = self.to_cell_types[0]
        io_cells = self.scaffold.get_cells_by_type(io_cell_type.name)

        # Get connection between molecular layer cells and Purkinje cells.
        molecular_cell_purkinje_connections = self.scaffold.get_connection_cache_by_cell_type(
            postsynaptic="purkinje_cell", presynaptic=molecular_cell_type.name
        )

        # Extract a list of cell types objects that are sources in the MLI to PC connections.
        # molecular_cell_purkinje_connections has the connection object from which we need to extract info as the first element
        print(molecular_cell_purkinje_connections)
        sources_mli_types = molecular_cell_purkinje_connections[0][0].from_cell_types
        # Associate an index to each MLI type which is connected to Purkinje cells
        index_mli_type = next(
            (
                index
                for (index, d) in enumerate(sources_mli_types)
                if d.name == molecular_cell_type.name
            ),
            None,
        )
        # second, third etc element in molecular_cell_purkinje_connections are the connection matrix for each element in from_cell_types
        molecular_cell_purkinje_matrix = molecular_cell_purkinje_connections[0][
            index_mli_type + 1
        ]

        io_cell_purkinje_connections = self.scaffold.get_connection_cache_by_cell_type(
            postsynaptic="purkinje_cell", presynaptic=io_cell_type.name
        )
        if len(io_cell_purkinje_connections[0]) < 2:
            # No IO to purkinje connections found. Do nothing.
            return
        io_cell_purkinje_matrix = io_cell_purkinje_connections[0][1]

        # Make a dictionary of which Purkinje cell is contacted by which molecular cells.
        purkinje_dict = {}
        for conn in range(len(molecular_cell_purkinje_matrix)):
            purkinje_id = molecular_cell_purkinje_matrix[conn][1]
            if not purkinje_id in purkinje_dict:
                purkinje_dict[purkinje_id] = []
            purkinje_dict[purkinje_id].append(molecular_cell_purkinje_matrix[conn][0])

        # Use the above dictionary to connect each IO cell to the molecular cells that
        # contact the Purkinje cells this IO cell contacts.
        io_molecular = []
        # Loop over all IO-Purkinje connections
        for io_conn in range(len(io_cell_purkinje_matrix)):
            io_id = io_cell_purkinje_matrix[io_conn][0]
            purkinje_id = io_cell_purkinje_matrix[io_conn][1]
            # No molecular cells contact this Purkinje cell
            if not purkinje_id in purkinje_dict:
                continue
            target_molecular_cells = purkinje_dict[purkinje_id]
            # Make a matrix that connects this IO cell to the target molecular cells
            matrix = np.column_stack(
                (np.repeat(io_id, len(target_molecular_cells)), target_molecular_cells,)
            )
            # Add the matrix to the output dataset.
            io_molecular.extend(matrix)
        # Store the connections.
        results = np.array(io_molecular)
        self.scaffold.connect_cells(self, results)


class TouchInformation:
    def __init__(
        self, from_cell_type, from_cell_compartments, to_cell_type, to_cell_compartments
    ):
        self.from_cell_type = from_cell_type
        self.from_cell_compartments = from_cell_compartments
        self.to_cell_type = to_cell_type
        self.to_cell_compartments = to_cell_compartments


class TouchDetector(ConnectionStrategy):
    """
        Connectivity based on intersection of detailed morphologies
    """

    casts = {
        "compartment_intersection_radius": float,
        "cell_intersection_radius": float,
        "synapses": DistributionConfiguration.cast,
        "allow_zero_synapses": bool,
    }

    defaults = {
        "cell_intersection_plane": "xyz",
        "compartment_intersection_plane": "xyz",
        "compartment_intersection_radius": 5.0,
        "synapses": DistributionConfiguration.cast(1),
        "allow_zero_synapses": False,
    }

    required = [
        "cell_intersection_plane",
        "compartment_intersection_plane",
        "compartment_intersection_radius",
    ]

    def validate(self):
        planes = ["xyz", "xy", "xz", "yz", "x", "y", "z"]
        assert_attr_in(
            self.__dict__,
            "cell_intersection_plane",
            planes,
            "connection_types.{}".format(self.name),
        )
        assert_attr_in(
            self.__dict__,
            "compartment_intersection_plane",
            planes,
            "connection_types.{}".format(self.name),
        )

    def connect(self):
        # Create a dictionary to cache loaded morphologies.
        self.morphology_cache = {}

        for from_cell_type_index in range(len(self.from_cell_types)):
            from_cell_type = self.from_cell_types[from_cell_type_index]
            from_cell_compartments = self.from_cell_compartments[from_cell_type_index]
            for to_cell_type_index in range(len(self.to_cell_types)):
                to_cell_type = self.to_cell_types[to_cell_type_index]
                to_cell_compartments = self.to_cell_compartments[to_cell_type_index]
                touch_info = TouchInformation(
                    from_cell_type,
                    from_cell_compartments,
                    to_cell_type,
                    to_cell_compartments,
                )
                # Intersect cells on the widest possible search radius.
                candidates = self.intersect_cells(touch_info)
                # Intersect cell compartments between matched cells.
                connections, morphology_names, compartments = self.intersect_compartments(
                    touch_info, candidates
                )
                # Connect the cells and store the morphologies and selected compartments that connect them.
                self.scaffold.connect_cells(
                    self,
                    connections,
                    morphologies=morphology_names,
                    compartments=compartments,
                )
        # Remove the morphology cache
        self.morphology_cache = None

    def intersect_cells(self, touch_info):
        from_cell_type = touch_info.from_cell_type
        to_cell_type = touch_info.to_cell_type
        cell_plane = self.cell_intersection_plane
        from_cell_tree = self.scaffold.trees.cells.get_planar_tree(
            from_cell_type.name, plane=cell_plane
        )
        to_cell_tree = self.scaffold.trees.cells.get_planar_tree(
            to_cell_type.name, plane=cell_plane
        )
        from_count = self.scaffold.get_placed_count(from_cell_type.name)
        to_count = self.scaffold.get_placed_count(to_cell_type.name)
        if hasattr(self, "cell_intersection_radius"):
            radius = self.cell_intersection_radius
        else:
            radius = self.get_search_radius(from_cell_type) + self.get_search_radius(
                to_cell_type
            )
        # TODO: Profile whether the reverse lookup with the smaller tree and then reversing the matches array
        # gains us any speed.
        if from_count < to_count:
            return to_cell_tree.query_radius(from_cell_tree.get_arrays()[0], radius)
        else:
            reversed_matches = from_cell_tree.query_radius(
                to_cell_tree.get_arrays()[0], radius
            )
            matches = [[] for _ in range(len(from_cell_tree.get_arrays()[0]))]
            for i in range(len(reversed_matches)):
                for match in reversed_matches[i]:
                    matches[match].append(i)
            return matches

    def intersect_compartments(self, touch_info, candidate_map):
        id_map_from = touch_info.from_cell_type.get_ids()
        id_map_to = touch_info.to_cell_type.get_ids()
        connected_cells = []
        morphology_names = []
        connected_compartments = []
        c_check = 0
        touching_cells = 0
        plots = 0
        for i in range(len(candidate_map)):
            from_id = id_map_from[i]
            touch_info.from_morphology = self.get_random_morphology(
                touch_info.from_cell_type
            )
            for j in candidate_map[i]:
                c_check += 1
                to_id = id_map_to[j]
                touch_info.to_morphology = self.get_random_morphology(
                    touch_info.to_cell_type
                )
                intersections = self.get_compartment_intersections(
                    touch_info, from_id, to_id
                )
                if len(intersections) > 0:
                    touching_cells += 1
                    number_of_synapses = max(
                        min(int(self.synapses.sample()), len(intersections)),
                        int(not self.allow_zero_synapses),
                    )
                    cell_connections = [
                        [from_id, to_id] for _ in range(number_of_synapses)
                    ]
                    compartment_connections = sample_elements(
                        intersections, k=number_of_synapses
                    )
                    connected_cells.extend(cell_connections)
                    connected_compartments.extend(compartment_connections)
                    # Pad the morphology names with the right names for the amount of compartment connections made
                    morphology_names.extend(
                        [
                            [
                                touch_info.from_morphology.morphology_name,
                                touch_info.to_morphology.morphology_name,
                            ]
                            for _ in range(len(compartment_connections))
                        ]
                    )
        if self.scaffold.configuration.verbosity > 1:
            print(
                "Checked {} candidate cell pairs from {} to {}".format(
                    c_check, touch_info.from_cell_type.name, touch_info.to_cell_type.name
                )
            )
            print(
                "Touch connection results: \n* Touching pairs: ",
                touching_cells,
                "\n* Synapses:",
                len(connected_compartments),
            )
        return (
            np.array(connected_cells, dtype=int),
            np.array(morphology_names, dtype=np.string_),
            np.array(connected_compartments, dtype=int),
        )

    def get_compartment_intersections(self, touch_info, from_cell_id, to_cell_id):
        from_cell_type = touch_info.from_cell_type
        to_cell_type = touch_info.to_cell_type
        from_morphology = touch_info.from_morphology
        to_morphology = touch_info.to_morphology
        from_pos = self.scaffold.get_cell_position(from_cell_id)
        to_pos = self.scaffold.get_cell_position(to_cell_id)
        query_points = (
            to_morphology.get_compartment_positions(types=touch_info.to_cell_compartments)
            + to_pos
            - from_pos
        )
        from_tree = from_morphology.get_compartment_tree(
            compartment_types=touch_info.from_cell_compartments
        )
        compartment_hits = from_tree.query_radius(
            query_points, self.compartment_intersection_radius
        )
        from_map = from_morphology.get_compartment_submask(
            compartment_types=touch_info.from_cell_compartments
        )
        to_map = to_morphology.get_compartment_submask(
            compartment_types=touch_info.to_cell_compartments
        )
        intersections = []
        for i in range(len(compartment_hits)):
            hits = compartment_hits[i]
            if len(hits) > 0:
                for j in range(len(hits)):
                    intersections.append([from_map[hits[j]], to_map[i]])
        return intersections

    def list_all_morphologies(self, cell_type):
        return cell_type.list_all_morphologies()

    def get_random_morphology(self, cell_type):
        """
            Return a morphology suited to represent a cell of the given `cell_type`.
        """
        available_morphologies = self.list_all_morphologies(cell_type)
        if len(available_morphologies) == 0:
            raise MissingMorphologyException(
                "Can't perform touch detection without detailed morphologies for {}".format(
                    cell_type.name
                )
            )
        m_name = random_element(available_morphologies)
        if m_name not in self.morphology_cache:
            mr = self.scaffold.morphology_repository
            self.morphology_cache[m_name] = mr.get_morphology(
                m_name, scaffold=self.scaffold
            )
        return self.morphology_cache[m_name]

    def get_all_morphologies(self, cell_type):
        all_morphologies = []
        for m_name in self.list_all_morphologies(cell_type):
            if m_name not in self.morphology_cache:
                mr = self.scaffold.morphology_repository
                self.morphology_cache[m_name] = mr.get_morphology(
                    m_name, scaffold=self.scaffold
                )
            all_morphologies.append(self.morphology_cache[m_name])
        return all_morphologies

    def get_search_radius(self, cell_type):
        morphologies = self.get_all_morphologies(cell_type)
        max_radius = 0.0
        for morphology in morphologies:
            max_radius = max(
                max_radius,
                np.max(
                    np.sqrt(
                        np.sum(
                            np.power(morphology.compartment_tree.get_arrays()[0], 2),
                            axis=1,
                        )
                    )
                ),
            )
        return max_radius


class SatelliteCommonPresynaptic(ConnectionStrategy):
    """
        Connectivity for satellite neurons (homologous to center neurons)
    """

    def validate(self):
        pass

    def connect(self):
        config = self.scaffold.configuration
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        after_connection = self.after
        after_cell_type = []
        for num_after in range(len(config.cell_types[to_type.name].placement.after)):
            after_cell_type.append(
                config.cell_types[to_type.name].placement.after[num_after]
            )
            after_connections = self.scaffold.cell_connections_by_tag[
                after_connection[num_after]
            ]
        if len(after_connections) == 0:
            return
        first_after = np.amin(after_connections[:, 1])
        to_cells = self.scaffold.get_cells_by_type(to_type.name)
        first_to = np.amin(to_cells)
        connections = np.column_stack(
            (after_connections[:, 0], after_connections[:, 1] - first_after + first_to)
        )
        self.scaffold.connect_cells(self, connections)


class AllToAll(ConnectionStrategy):
    """
        All to all connectivity between two neural populations
    """

    def validate(self):
        pass

    def connect(self):
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        from_cells = self.scaffold.get_cells_by_type(from_type.name)
        to_cells = self.scaffold.get_cells_by_type(to_type.name)
        l = len(to_cells)
        connections = np.empty([len(from_cells) * l, 2])
        to_cell_ids = to_cells[:, 0]
        for i, from_cell in enumerate(from_cells[:, 0]):
            connections[range(i * l, (i + 1) * l), 0] = from_cell
            connections[range(i * l, (i + 1) * l), 1] = to_cell_ids
        self.scaffold.connect_cells(self, connections)


class ConnectomeMossyGlomerulus(ConnectionStrategy):
    """
        Implementation for the connections between mossy fibers and glomeruli.
        The connectivity is somatotopic and
    """

    def validate(self):
        pass

    def connect(self):
        def probability_mapping(input, center, std):
            # input: input array that has to be transformed
            # center: center of the sigmoid
            # std: value at which the sigmoid reaches the 54% of its value
            output = np.empty(input.size, dtype=float)
            input_rect = np.fabs(input - center)
            output[np.where(input <= center)] = (
                0.5 + 0.5 * (input[np.where(input <= center)]) / center
            )
            output[np.where(input > center)] = 2.0 * (
                1.0
                - 1.0
                / (1.0 + np.exp(-input_rect[np.where(input > center)] * (1.0 / std)))
            )
            return output

        def compute_likelihood(x, z, gloms):
            # Based on the distance between the x and z position of each
            # MF and the x z positions of the glomeruli
            # the likelihood of a glomerulus to belong to the MF
            # is computed
            dist_x = np.fabs(gloms[:, 0] - x)
            dist_z = np.fabs(gloms[:, 1] - z)

            prob_x = probability_mapping(
                dist_x, center=30.0, std=3.0
            )  # As in Sultan, 2001 for the parasagittal axis
            prob_z = probability_mapping(
                dist_z, center=10.0, std=1.0
            )  # As in Sultan, 2001 for the mediolateral axis

            probabilities = prob_x * prob_z
            return probabilities

        # Source and target neurons are extracted
        mossy_cell_type = self.from_cell_types[0]
        glomerulus_cell_type = self.to_cell_types[0]
        mossy = self.scaffold.entities_by_type[mossy_cell_type.name].astype(int)
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        # Number of MFs placed and ID of the first MF
        MF_num = np.shape(mossy)[0]
        First_MF = np.min(mossy)

        # Glom x, y and ID
        Glom_xzID = glomeruli[:, [2, 4, 0]]
        total_glom = np.shape(Glom_xzID)[0]

        # Boundaries of X and Z space for glomeruli
        BoundsX = np.array([np.min(Glom_xzID[:, 0]), np.max(Glom_xzID[:, 0])])
        BoundsZ = np.array([np.min(Glom_xzID[:, 1]), np.max(Glom_xzID[:, 1])])

        # Computation of how many MFs do we need to "place" for the two axes
        XZ_Area = (BoundsX[1] - BoundsX[0]) * (BoundsZ[1] - BoundsZ[0])
        MF_per_Area = MF_num / XZ_Area
        MF_per_X = np.ceil((BoundsX[1] - BoundsX[0]) * np.sqrt(MF_per_Area)).astype(int)
        MF_per_Z = np.ceil((BoundsZ[1] - BoundsZ[0]) * np.sqrt(MF_per_Area)).astype(int)

        # Create uniform grid in the X-Z plane
        MF_X = np.linspace(BoundsX[0], BoundsX[1], num=MF_per_X)
        MF_Z = np.linspace(BoundsZ[0], BoundsZ[1], num=MF_per_Z)
        xv, zv = np.meshgrid(MF_X, MF_Z, sparse=False, indexing="ij")
        xv = xv.flatten()
        zv = zv.flatten()

        # Limit the number of MFs (xv and zv) to MF_num
        if np.size(xv) > MF_num:
            delete_points = np.random.randint(0, np.size(xv), size=np.size(xv) - MF_num)
            xv = np.delete(xv, delete_points)
            zv = np.delete(zv, delete_points)

        # labels store the assigned MF to each glomerulus
        labels = -1 * np.ones(np.shape(Glom_xzID)[0], dtype=int)
        best_glom = -1 * np.ones(MF_num * MF_num, dtype=int)
        best_prob = np.zeros(MF_num, dtype=float)
        min_glom = np.min(Glom_xzID[:, 2]).astype(int)

        # This loop iterates associating at each time one glomeurlus to the MF
        # that has the maximum likelihood to be connected to it
        while np.shape(Glom_xzID)[0] > 0:
            # Every time the array is shuffled to avoid bias toward the first glumeruli in the list
            np.random.shuffle(Glom_xzID)
            # For each MF, the highest probability (best_prob) and the corresponding glumerulus (best_blom)
            # are computed
            for i in range(MF_num):
                probabilities = compute_likelihood(xv[i], zv[i], Glom_xzID)
                best_glom[i] = np.argmax(probabilities)
                best_prob[i] = np.max(probabilities)
            # We select the best glomerulus among the best ones for each MF
            highest_glom_MF = np.argmax(best_prob)
            # The label of that glomerulus is assigned
            labels[
                int(Glom_xzID[best_glom[highest_glom_MF], 2]) - min_glom
            ] = highest_glom_MF
            # That glomerulus is deleted from the list
            Glom_xzID = np.delete(Glom_xzID, best_glom[highest_glom_MF], axis=0)
            self.scaffold.report(
                "Associated "
                + str(int(100 * (1 - np.shape(Glom_xzID)[0] / total_glom)))
                + "% glomeruli",
                ongoing=True,
                level=3,
            )
            if np.shape(Glom_xzID)[0] == 0:
                break
        # Labels range from 0 to MF_num, while they should range from First_MF to First_MF+MF_num
        labels += First_MF
        connections = np.column_stack((labels, glomeruli[:, 0]))
        self.scaffold.connect_cells(self, connections)
