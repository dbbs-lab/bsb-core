import numpy as np
from ..strategy import ConnectionStrategy
from ...helpers import DistributionConfiguration
from ...functions import get_distances
from scipy.stats.distributions import truncexpon


class ConnectomeGlomerulusGolgi(ConnectionStrategy):
    """
        Legacy implementation for the connections between Golgi cells and glomeruli.
    """

    defaults = {"detailed": False, "contacts": DistributionConfiguration.cast(1)}
    casts = {"detailed": bool, "contacts": DistributionConfiguration.cast}

    def validate(self):
        if self.detailed:
            morphologies = self.to_cell_types[0].list_all_morphologies()
            if not morphologies:
                raise ConfigurationError(
                    "Can't create detailed glomerulus to Golgi connections without any morphologies for the Golgi cell."
                )
            elif len(morphologies) > 1:
                raise NotImplementedError(
                    "Detailed glomerulus to Golgi connections can only be made for a single morphology."
                    + " (Requires the selection of morphologies to be moved from the connection module to the placement module)"
                )
            mr = self.scaffold.morphology_repository
            morphology = mr.get_morphology(morphologies[0])
            self.dendritic_compartments = morphology.get_compartments(["dendrites"])
            self.morphology = morphology

    def connect(self):
        # Gather information for the legacy code block below.
        glomerulus_cell_type = self.from_cell_types[0]
        golgi_cell_type = self.to_cell_types[0]
        glomeruli = self.scaffold.cells_by_type[glomerulus_cell_type.name]
        golgis = self.scaffold.cells_by_type[golgi_cell_type.name]
        first_glomerulus = int(glomeruli[0, 0])
        r_goc_vol = golgi_cell_type.morphology.dendrite_radius
        if self.detailed:
            compartments = np.zeros((0, 2))
            comps = self.dendritic_compartments
            total_compartments = len(comps)

        def connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol):
            nonlocal compartments
            glom_bd = np.zeros((0, 2))
            glom_x = glomeruli[:, 2]
            glom_y = glomeruli[:, 3]
            glom_z = glomeruli[:, 4]
            # If synaptic contacts need to be made we use this exponential distribution
            # to pick the closer by compartments.
            exp_dist = truncexpon(b=5, scale=0.03)
            # for all Golgi cells: calculate which glomeruli fall into the volume of GoC
            # basolateral dendrites, then choose 40 of them for the connection and delete
            # them from successive computations, since 1 axon is connected to 1 GoC
            for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in golgicells:
                # Geometric constraints: glom less than `r_goc_vol` away from golgi and
                # golgi cell soma above glom.
                volume_matrix = (
                    ((glom_x - golgi_x) ** 2)
                    + ((glom_y - golgi_y) ** 2)
                    + ((glom_z - golgi_z) ** 2)
                    - (r_goc_vol ** 2)
                ).__le__(0) & glom_y.__le__(golgi_y)
                # finds indexes of granules that can potentially be connected
                good_gloms = np.where(volume_matrix == True)[0]
                # Translate local id to simulation id
                connected_gloms = good_gloms + first_glomerulus
                if self.detailed:
                    # Draw a sample from the configured contacts distribution for each
                    # connected glomerulus.
                    samples = [int(x) for x in self.contacts.draw(len(connected_gloms))]
                    # The total synaptic contacts is the sum of the contacts with each
                    # glomerulus.
                    total_contacts = sum(samples)
                    # Compose an empty connection matrix to hold the connection records
                    matrix = np.zeros((total_contacts, 2))
                    # Holds the connection_compartments for this golgi cell
                    gg_comps = np.zeros(matrix.shape)
                    # Draw rolls from the exponential distribution equal to the total amount
                    # of synaptic contacts to be made between this Golgi cell and all its
                    # glomeruli.
                    rolls = exp_dist.rvs(size=total_contacts)

                    pointer = 0
                    for i, cg in enumerate(connected_gloms):
                        sample = samples[i]
                        _end = pointer + sample
                        # Fill in `sample` records of the connection matrix
                        matrix[pointer:_end, 0] = cg
                        matrix[pointer:_end, 1] = golgi_id
                        gid = good_gloms[i]
                        # Get the distance sorted compartment indices
                        d = np.array(
                            get_distances(
                                [c.start for c in comps],
                                [
                                    glom_x[gid] - golgi_x,
                                    glom_y[gid] - golgi_y,
                                    glom_z[gid] - golgi_z,
                                ],
                            )
                        )
                        d_comps = np.argsort(d)
                        # Pick compartments according to a exponential distribution mapped
                        # through the distance indices: high chance to pick closeby comps.
                        gg_comps[pointer:_end, 1] = [
                            comps[d_comps[int(k * total_compartments)]].id
                            for k in rolls[pointer:_end]
                        ]
                        pointer = _end
                    # Stack the connection matrix
                    glom_bd = np.vstack((glom_bd, matrix))
                    # Stack the compartment matrix
                    compartments = np.vstack((compartments, gg_comps))
                else:
                    matrix = np.zeros((len(connected_gloms), 2))
                    matrix[:, 0] = connected_gloms  # from cell
                    matrix[:, 1] = golgi_id  # to cell
                    glom_bd = np.vstack((glom_bd, matrix))

            return glom_bd

        connectome = connectome_glom_goc(first_glomerulus, glomeruli, golgis, r_goc_vol)
        if self.detailed:
            morphologies = np.zeros(connectome.shape)
            morphology_map = [self.morphology.morphology_name]
            self.scaffold.connect_cells(
                self,
                connectome,
                compartments=compartments,
                morphologies=morphologies,
                morpho_map=morphology_map,
            )
        else:
            self.scaffold.connect_cells(self, connectome)
