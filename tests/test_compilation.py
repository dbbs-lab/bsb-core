import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.models import Layer, CellType

class TestSingleTypeCompilation(unittest.TestCase):
    '''
    Check if the scaffold can create a single cell type.
    '''

    @classmethod
    def setUpClass(self):
        super(TestSingleTypeCompilation, self).setUpClass()
        config = JSONConfig(file="configs/test_single_neuron.json")
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()

    def test_placement_statistics(self):
        self.assertEqual(self.scaffold.statistics.cells_placed["test_cell"], 4)

    def test_network_cache(self):
        pass
        # TODO: Implement a check that the network cache contains the right amount of placed cells

    def test_hdf5_cells(self):
        pass
        # TODO: Implement a check that the hdf5 file contains the right datasets under 'cells'

class TestPlacement(unittest.TestCase):
    '''
    Check if the placement of all cell types is correct
    '''

    @classmethod
    def setUpClass(self):
        super(TestPlacement, self).setUpClass()
        config = JSONConfig(file="../scaffold/configurations/mouse_cerebellum.json")
        self.scaffold = Scaffold(config)

    def test_purkinje(self):
        import scipy.spatial.distance as dist
        config = self.scaffold.configuration
        layer = config.layers['purkinje_layer']
        pc = config.cell_types['purkinje_cell']
        self.scaffold.reset_network_cache()
        pc.placement.place(pc)
        pcCount = self.scaffold.cells_by_type['purkinje_cell'].shape[0]
        density = pcCount / layer.width / layer.depth
        pc_pos = self.scaffold.cells_by_type['purkinje_cell'][:, [2,3,4]]
        Dist2D = dist.pdist(np.column_stack((pc_pos[:,0],pc_pos[:,2])), 'euclidean')
        overlapSomata = np.where(Dist2D < 80/100*pc.placement.soma_radius)[0] #
        Dist1Dsqr=np.zeros(( pc_pos.shape[0],pc_pos.shape[0],2))
        Dist1Dsqr[:,:,0] = dist.squareform(dist.pdist(pc_pos[:,[0]], 'euclidean'))
        Dist1Dsqr[:,:,1] = dist.squareform(dist.pdist(pc_pos[:,[2]], 'euclidean'))
        overlapDend_whichPairs = np.where ( np.logical_and (Dist1Dsqr[:,:,0] <80/100*pc.placement.extension_x, Dist1Dsqr[:,:,1]<80/100*pc.placement.extension_z ) )
        overlapDend_whichPairs = np.column_stack((overlapDend_whichPairs[0],overlapDend_whichPairs[1]))
        overlapDend_whichPairs = overlapDend_whichPairs[np.where (overlapDend_whichPairs[:,0]!=overlapDend_whichPairs[:,1])[0],:]

        # Asserts
        self.assertAlmostEqual(overlapSomata.shape[0],0, delta=pcCount*1/100)
        self.assertAlmostEqual(overlapDend_whichPairs.shape[0]/2,0, delta=pcCount*4/100)
