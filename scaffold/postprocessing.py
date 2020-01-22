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
        for neurons_2b_labeled in [
            "purkinje_cell",
            "dcn_cell",
            "dcn_interneuron",
            "io_cell",
        ]:
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
            self.scaffold.label_cells(
                self.scaffold.get_cells_by_type(neurons_2b_labeled)[index_pos, 0],
                label="Microzone_positive",
            )
            self.scaffold.label_cells(
                self.scaffold.get_cells_by_type(neurons_2b_labeled)[index_neg, 0],
                label="Microzone_negative",
            )


def get_parallel_fiber_heights(scaffold, granule_geometry, granules):
    parallel_fibers = np.zeros((len(granules), 2))
    pf_height = granule_geometry.pf_height
    pf_height_sd = granule_geometry.pf_height_sd
    molecular_layer = scaffold.configuration.get_layer(name="molecular_layer")
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
    return parallel_fibers


def get_dcn_rotations(dcn_matrix):
    """
        Create a matrix of planes tilted between -45° and 45°,
        storing id and the planar coefficients a, b, c and d for each DCN cell
    """
    dend_tree_coeff = np.zeros((dcn_matrix.shape[0], 4))
    for i in range(len(dcn_matrix)):
        # Make the planar coefficients a, b and c.
        dend_tree_coeff[i] = np.random.rand(4) * 2.0 - 1.0
        # Calculate the last planar coefficient d from ax + by + cz - d = 0
        # => d = - (ax + by + cz)
        dend_tree_coeff[i, 3] = -np.sum(dend_tree_coeff[i, 0:2] * dcn_matrix[i, 2:4])
    # Compose the matrix
    return np.column_stack((dcn_matrix[:, 0], dend_tree_coeff))
