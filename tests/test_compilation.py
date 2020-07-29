import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig
from bsb.models import Layer, CellType
from bsb.placement import Satellite
from test_setup import get_test_network


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


single_neuron_config = relative_to_tests_folder("configs/test_single_neuron.json")


class TestSingleTypeCompilation(unittest.TestCase):
    """
        Check if the scaffold can create a single cell type.
    """

    @classmethod
    def setUpClass(self):
        super(TestSingleTypeCompilation, self).setUpClass()
        config = JSONConfig(file=single_neuron_config)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()

    def test_placement_statistics(self):
        self.assertEqual(self.scaffold.statistics.cells_placed["test_cell"], 4)
        self.assertEqual(self.scaffold.get_cell_total(), 4)

    def test_network_cache(self):
        pass
        # TODO: Implement a check that the network cache contains the right amount of placed cells

    def test_hdf5_cells(self):
        pass
        # TODO: Implement a check that the hdf5 file contains the right datasets under 'cells'

    def test_from_hdf5(self):
        scaffold_copy = from_hdf5(self.scaffold.output_formatter.file)
        for key in self.scaffold.statistics.cells_placed:
            self.assertEqual(
                scaffold_copy.statistics.cells_placed[key],
                self.scaffold.statistics.cells_placed[key],
            )
        self.assertEqual(scaffold_copy.get_cell_total(), 4)
        self.assertRaises(OSError, from_hdf5, "doesntexist")


_using_morphologies = True


@unittest.skipIf(_using_morphologies, "Morphologies are used for the connectivity")
class TestPlacement(unittest.TestCase):
    """
        Check if the placement of all cell types is correct
    """

    @classmethod
    def setUpClass(self):
        super(TestPlacement, self).setUpClass()
        self.scaffold = get_test_network(200, 200)

    def test_bounds(self):
        # Create different sized test networks
        for dimensions in [[100, 100], [200, 200]]:
            scaffold = get_test_network(*dimensions)
            # Get all non-entity cell types.
            for cell_type in scaffold.get_cell_types(entities=False):
                # Start an out-of-bounds check.
                with self.subTest(
                    cell_type=cell_type.name,
                    dimensions=dimensions,
                    placement=cell_type.placement.__class__.__name__,
                ):
                    # Treat satellites differently because they can belong to
                    # multiple layers of their planet cell types.
                    if isinstance(cell_type.placement, Satellite):
                        self._test_satellite_bounds(cell_type)
                        continue
                    # Get layer bounds and cell positions.
                    layer = cell_type.placement.layer_instance
                    positions = scaffold.get_cells_by_type(cell_type.name)[:, 2:5]
                    min = layer.origin
                    max = layer.origin + layer.dimensions
                    # If any element in the positional data is smaller than its
                    # corresponding element in `min` or larger than that in `max`,
                    # np.where will add its index to an array which needs to be empty
                    # for the test to pass.
                    i = len(np.where(((positions < min) | (positions > max)))[0])
                    self.assertEqual(i, 0, "Cells placed out of bounds.")

    def _test_satellite_bounds(self, cell_type):
        # Use the after array to get all planet cell type layers.
        after = cell_type.placement.after
        planet_cell_types = [self.scaffold.get_cell_type(n) for n in after]
        layers = [
            planet_cell_type.placement.layer_instance
            for planet_cell_type in planet_cell_types
        ]
        # Get satellite positions
        positions = self.scaffold.get_cells_by_type(cell_type.name)[:, 2:5]
        in_bounds = set()
        # Check layer per layer whether the satellites are in a layer Update a set per
        # layer to keep track of all satellites that belong to at least one planet layer
        for layer in layers:
            min = layer.origin
            max = layer.origin + layer.dimensions
            in_layer_bounds = np.where(
                np.all((positions > min) & (positions < max), axis=1)
            )[0]
            in_bounds.update(in_layer_bounds)
        # Verify all satellite cells belong to at least one planet layer.
        out_of_bounds = len(positions) - len(in_bounds)
        message = str(out_of_bounds) + " cells placed out of bounds"
        self.assertEqual(out_of_bounds, 0, message)

    def test_purkinje(self):
        import scipy.spatial.distance as dist

        config = self.scaffold.configuration
        layer = config.layers["purkinje_layer"]
        pc = config.cell_types["purkinje_cell"]
        self.scaffold.reset_network_cache()
        pc.placement.place()
        pcCount = self.scaffold.cells_by_type["purkinje_cell"].shape[0]
        density = pcCount / layer.width / layer.depth
        pc_pos = self.scaffold.cells_by_type["purkinje_cell"][:, [2, 3, 4]]
        Dist2D = dist.pdist(np.column_stack((pc_pos[:, 0], pc_pos[:, 2])), "euclidean")
        overlapSomata = np.where(Dist2D < 80 / 100 * pc.placement.soma_radius)[0]  #
        Dist1Dsqr = np.zeros((pc_pos.shape[0], pc_pos.shape[0], 2))
        Dist1Dsqr[:, :, 0] = dist.squareform(dist.pdist(pc_pos[:, [0]], "euclidean"))
        Dist1Dsqr[:, :, 1] = dist.squareform(dist.pdist(pc_pos[:, [2]], "euclidean"))
        overlapDend_whichPairs = np.where(
            np.logical_and(
                Dist1Dsqr[:, :, 0] < 80 / 100 * pc.placement.extension_x,
                Dist1Dsqr[:, :, 1] < 80 / 100 * pc.placement.extension_z,
            )
        )
        overlapDend_whichPairs = np.column_stack(
            (overlapDend_whichPairs[0], overlapDend_whichPairs[1])
        )
        overlapDend_whichPairs = overlapDend_whichPairs[
            np.where(overlapDend_whichPairs[:, 0] != overlapDend_whichPairs[:, 1])[0], :
        ]

        # Asserts
        self.assertAlmostEqual(overlapSomata.shape[0], 0, delta=pcCount * 1 / 100)
        self.assertAlmostEqual(
            overlapDend_whichPairs.shape[0] / 2, 0, delta=pcCount * 4 / 100
        )
