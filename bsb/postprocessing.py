from .helpers import ConfigurableClass
from .reporting import report, warn
from scipy.stats import truncnorm
import numpy as np
from .exceptions import *


class PostProcessingHook(ConfigurableClass):
    def validate(self):
        pass

    def after_placement(self):
        raise NotImplementedError(
            "`after_placement` hook not defined on " + self.__class__.__name__
        )

    def after_connectivity(self):
        raise NotImplementedError(
            "`after_connectivity` hook not defined on " + self.__class__.__name__
        )


class LabelMicrozones(PostProcessingHook):
    def after_placement(self):
        # Divide the volume into two sub-parts (one positive and one negative)
        for neurons_2b_labeled in self.targets:
            ids = self.scaffold.get_cells_by_type(neurons_2b_labeled)[:, 0]
            zeds = self.scaffold.get_cells_by_type(neurons_2b_labeled)[:, 4]
            z_sep = np.median(zeds)
            index_pos = np.where(zeds >= z_sep)[0]
            index_neg = np.where(zeds < z_sep)[0]
            report(
                neurons_2b_labeled
                + " divided into microzones: {} positive, {} negative".format(
                    index_pos.shape[0], index_neg.shape[0]
                ),
                level=3,
            )

            labels = {
                "microzone-positive": ids[index_pos],
                "microzone-negative": ids[index_neg],
            }

            self.scaffold.label_cells(
                ids[index_pos],
                label="microzone-positive",
            )
            self.scaffold.label_cells(
                ids[index_neg],
                label="microzone-negative",
            )

            self.label_satellites(neurons_2b_labeled, labels)

    def label_satellites(self, planet_type, labels):
        for possible_satellites in self.scaffold.get_cell_types():
            # Find all cell types that specify this type as their planet type
            if (
                hasattr(possible_satellites.placement, "planet_types")
                and planet_type in possible_satellites.placement.planet_types
            ):
                # Get the IDs of this sattelite cell type.
                satellites = self.scaffold.get_cells_by_type(possible_satellites.name)[
                    :, 0
                ]
                # Retrieve the planet map for this satellite type. A planet map is an
                # array that lists the planet for each satellite. `sattelite_map[n]`` will
                # hold the planet ID for sattelite `n`, where `n` the index of the
                # satellites in their cell type, not their scaffold ID.
                satellite_map = self.scaffold._planets[possible_satellites.name].copy()
                # Create counters for each label for the report below
                satellite_label_count = {l: 0 for l in labels.keys()}
                # Iterate each label to check for any planets with that label, and label
                # their sattelite with the same label. After iterating all labels, each
                # satellite should have the same labels as their planet.
                for label, labelled_cells in labels.items():
                    for i, satellite in enumerate(satellites):
                        planet = satellite_map[i]
                        if planet in labelled_cells:
                            self.scaffold.label_cells([satellite], label=label)
                            # Increase the counter of this label
                            satellite_label_count[label] += 1
                if sum(satellite_label_count.values()) > 0:
                    # Report how many labels have been applied to which cell type.
                    report(
                        "{} are satellites of {} and have been labelled as: {}".format(
                            possible_satellites.name,
                            planet_type,
                            ", ".join(
                                map(
                                    lambda label, count: str(count) + " " + label,
                                    satellite_label_count.keys(),
                                    satellite_label_count.values(),
                                )
                            ),
                        ),
                        level=3,
                    )


class AscendingAxonLengths(PostProcessingHook):
    def after_placement(self):
        granule_type = self.scaffold.get_cell_type("granule_cell")
        granules = self.scaffold.get_cells_by_type(granule_type.name)
        granule_geometry = granule_type.morphology
        parallel_fibers = np.zeros((len(granules), 2))
        pf_height = granule_geometry.pf_height
        pf_height_sd = granule_geometry.pf_height_sd
        molecular_layer = self.scaffold.configuration.get_layer(name="molecular_layer")
        floor_ml = molecular_layer.Y
        roof_ml = floor_ml + molecular_layer.height  # Roof of the molecular layer

        for idx, granule in enumerate(granules):
            granule_y = granule[3]
            # Determine min and max height so that the parallel fiber is inside of the molecular layer
            pf_height_min = floor_ml - granule_y
            pf_height_max = roof_ml - granule_y
            # Determine the shape parameters a and b of the truncated normal distribution.
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            a, b = (
                (pf_height_min - pf_height) / pf_height_sd,
                (pf_height_max - pf_height) / pf_height_sd,
            )
            # Draw a sample for the parallel fiber height from a truncated normal distribution
            # with sd `pf_height_sd` and mean `pf_height`, truncated by the molecular layer bounds.
            parallel_fibers[idx, 1] = (
                truncnorm.rvs(a, b, size=1) * pf_height_sd + pf_height
            )  # Height
            parallel_fibers[idx, 0] = granule[0]  # ID
        self.scaffold.append_dset("cells/ascending_axon_lengths", data=parallel_fibers)


class DCNRotations(PostProcessingHook):
    """
    Create a matrix of planes tilted between -45° and 45°,
    storing id and the planar coefficients a, b, c and d for each DCN cell
    """

    def after_placement(self):
        dcn_matrix = self.scaffold.get_cells_by_type("dcn_cell")
        dend_tree_coeff = np.zeros((dcn_matrix.shape[0], 4))
        for i in range(len(dcn_matrix)):
            # Make the planar coefficients a, b and c.
            dend_tree_coeff[i] = np.random.rand(4) * 2.0 - 1.0
            # Calculate the last planar coefficient d from ax + by + cz - d = 0
            # => d = - (ax + by + cz)
            dend_tree_coeff[i, 3] = -np.sum(dend_tree_coeff[i, 0:2] * dcn_matrix[i, 2:4])
        # Compose the matrix
        matrix = np.column_stack((dcn_matrix[:, 0], dend_tree_coeff))
        # Save the matrix
        self.scaffold.append_dset("cells/dcn_orientations", data=matrix)


class SpoofDetails(PostProcessingHook):
    """
    Create fake morphological intersections between already connected non-detailed
    connection types.
    """

    casts = {"presynaptic": str, "postsynaptic": str}

    def after_connectivity(self):
        # Check which connection types exist between the pre- and postsynaptic types.
        connection_results = self.scaffold.get_connection_cache_by_cell_type(
            presynaptic=self.presynaptic, postsynaptic=self.postsynaptic
        )
        # Iterate over each involved connectivity matrix
        for connection_result in connection_results:
            connection_type = connection_result[0]
            for connectivity_matrix in connection_result[1:]:
                # Spoof details (morphology & section intersection) between the
                # non-detailed connections in the connectivity matrix.
                self.spoof_connections(connection_type, connectivity_matrix)

    def spoof_connections(self, connection_type, connectivity_matrix):
        from_type = connection_type.from_cell_types[0]
        to_type = connection_type.to_cell_types[0]
        from_relay = False
        to_relay = False
        # Check whether any of the types are relays or entities.
        if from_type.relay or from_type.entity:
            from_relay = True
            if to_type.relay or to_type.entity:
                raise MorphologyError(
                    "Can't spoof detailed connections between 2 relay or entity cell types."
                )
        elif to_type.relay or to_type.entity:
            to_relay = True
        # If they're not relays or entities, load their morphologies
        if not from_relay:
            from_morphologies = from_type.list_all_morphologies()
            if len(from_morphologies) == 0:
                raise MorphologyDataError(
                    "Can't spoof detailed connection without morphologies for '{}'".format(
                        from_type.name
                    )
                )
        if not to_relay:
            to_morphologies = to_type.list_all_morphologies()
            if len(to_morphologies) == 0:
                raise MorphologyDataError(
                    "Can't spoof detailed connection without morphologies for '{}'".format(
                        to_type.name
                    )
                )
        # If they are entities or relays, steal the first morphology of the other cell type.
        # Under no circumstances should entities or relays be represented as actual
        # morphologies, so this should not matter: the data just needs to be spoofed for
        # other parts of the scaffold to function.
        if from_relay:
            from_morphologies = [to_morphologies[0]]
        if to_relay:
            to_morphologies = [from_morphologies[0]]

        # Use only the first morphology for spoofing.
        # At a later point which morphology belongs to which cell should be decided
        # as a property of the cell and not the connection.
        # At that point we can spoof the same morphologies to the opposing relay type.
        #
        # The left column will be the first from_morphology (0) and the right column
        # will be the first to_morphology (1)
        morphologies = np.column_stack(
            (
                np.zeros(
                    (
                        len(
                            connectivity_matrix,
                        )
                    )
                ),
                np.ones(
                    (
                        len(
                            connectivity_matrix,
                        )
                    )
                ),
            )
        )
        # Generate the map
        morpho_map = [from_morphologies[0], to_morphologies[0]]
        from_m = self.scaffold.morphology_repository.get_morphology(from_morphologies[0])
        to_m = self.scaffold.morphology_repository.get_morphology(to_morphologies[0])
        # Select random axons and dendrites to connect
        axons = np.array(from_m.get_compartment_submask(["axon"]))
        dendrites = np.array(from_m.get_compartment_submask(["dendrites"]))
        compartments = np.column_stack(
            (
                axons[np.random.randint(0, len(axons), len(connectivity_matrix))],
                dendrites[np.random.randint(0, len(dendrites), len(connectivity_matrix))],
            )
        )
        # Erase previous connection data so that `.connect_cells` can overwrite it.
        self.scaffold.cell_connections_by_tag[connection_type.name] = np.empty((0, 2))
        # Write the new spoofed connection data
        self.scaffold.connect_cells(
            connection_type,
            connectivity_matrix,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )
        report(
            "Spoofed details of {} connections between {} and {}".format(
                len(connectivity_matrix),
                connection_type.from_cell_types[0].name,
                connection_type.to_cell_types[0].name,
            ),
            level=2,
        )


class GolgiAxonFix(PostProcessingHook):
    # Replaces the presynaptic compartment IDs of all Golgi cells with the soma compartment
    def after_connectivity(self):
        for n, ct in self.scaffold.configuration.connection_types.items():
            if ct.from_cell_types[0].name == "golgi_cell":
                for tag in ct.tags:
                    compartment_matrix = self.scaffold.connection_compartments[tag]
                    compartment_matrix[:, 0] = np.zeros(len(compartment_matrix))
