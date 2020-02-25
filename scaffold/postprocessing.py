from .helpers import ConfigurableClass
from scipy.stats import truncnorm
import numpy as np


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
            zeds = self.scaffold.get_cells_by_type(neurons_2b_labeled)[:, 4]
            z_sep = np.median(zeds)
            index_pos = np.where(zeds >= z_sep)[0]
            index_neg = np.where(zeds < z_sep)[0]
            self.scaffold.report(
                neurons_2b_labeled
                + " divided into microzones: {} positive, {} negative".format(
                    index_pos.shape[0], index_neg.shape[0]
                ),
                level=3,
            )

            labels = {
                "microzone-positive": self.scaffold.get_cells_by_type(neurons_2b_labeled)[
                    index_pos, 0
                ],
                "microzone-negative": self.scaffold.get_cells_by_type(neurons_2b_labeled)[
                    index_neg, 0
                ],
            }

            self.scaffold.label_cells(
                self.scaffold.get_cells_by_type(neurons_2b_labeled)[index_pos, 0],
                label="microzone-positive",
            )
            self.scaffold.label_cells(
                self.scaffold.get_cells_by_type(neurons_2b_labeled)[index_neg, 0],
                label="microzone-negative",
            )

            self.label_satellites(neurons_2b_labeled, labels)

    def label_satellites(self, planet_type, labels):
        for possible_satellites in self.scaffold.get_cell_types():
            if (
                hasattr(possible_satellites.placement, "planet_types")
                and planet_type in possible_satellites.placement.planet_types
            ):
                satellites = self.scaffold.get_cells_by_type(possible_satellites.name)[
                    :, 0
                ]
                satellite_map = self.scaffold._planets[possible_satellites.name].copy()
                lab_pos = lab_neg = 0
                for i, satellite in enumerate(satellites):
                    planet = satellite_map[i]
                    if planet in labels["microzone-positive"]:
                        self.scaffold.label_cells(
                            np.array([satellite]), label="microzone-positive"
                        )
                        lab_pos += 1
                    else:
                        self.scaffold.label_cells(
                            np.array([satellite]), label="microzone-negative"
                        )
                        lab_neg += 1
                num_labelled_sat = lab_pos + lab_neg
                if num_labelled_sat > 0:
                    self.scaffold.report(
                        "{} are satellites of {} and have been labelled as: {} positive, {} negative".format(
                            possible_satellites.name, planet_type, lab_pos, lab_neg
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


class SpoofGlomerulusGranuleDetailed(PostProcessingHook):
    """
        Create a detailed intersection for each glomerulus to granule connection.
        Empty section data is created for the glomerulus as it has no morphology, and a
        random granule dendrite is selected.
    """

    def after_connectivity(self):
        connection_results = self.scaffold.get_connection_cache_by_cell_type(
            presynaptic="glomerulus", postsynaptic="granule_cell"
        )
        connectivity_matrix = connection_results[0][1]
        ctype = connection_results[0][0]
        # Erase previous data so that .connect_cells can be used again
        self.scaffold.cell_connections_by_tag["glomerulus_to_granule"] = np.empty((0, 2))
        # Use a hardcoded granule morphology for both
        morphologies = np.zeros((len(connectivity_matrix), 2))
        morpho_map = ["granule_dbbs"]
        m = self.scaffold.morphology_repository.get_morphology("granule_dbbs")
        # Select random dendrites
        dendrites = np.array([c.id for c in m.compartments if c.type == 3])
        compartments = np.column_stack(
            (
                np.zeros(len(connectivity_matrix)),
                dendrites[np.random.randint(0, len(dendrites), len(connectivity_matrix))],
            )
        )
        self.scaffold.connect_cells(
            ctype,
            connectivity_matrix,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )
