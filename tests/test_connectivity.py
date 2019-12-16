import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.scaffold import Scaffold
from scaffold.config import JSONConfig, from_hdf5
from scaffold.models import Layer, CellType


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


config_connectivity = relative_to_tests_folder("../scaffold/configurations/mouse_cerebellum.json")


class TestConnectivity(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(TestConnectivity, self).setUpClass()
        config = JSONConfig(file=config_connectivity)
        config.verbosity = 3
        config.resize(X=200, Z=200)
        self.scaffold = Scaffold(config)
        self.scaffold.compile_network()
        self.nest_adapter = self.scaffold.get_simulation("FCN_2019")
        self.scaffold.run_simulation("FCN_2019")

    def test_MF_Glom(self):
        mossy = self.nest_adapter.entities['mossy_fibers'].nest_identifiers
        glom = self.nest_adapter.cell_models['glomerulus'].nest_identifiers
        mf_glom = np.array(self.nest_adapter.nest.GetConnections(mossy, glom))
        hist_mf = np.histogram(mf_glom[:, 0], bins=np.append(mossy, np.max(mossy)+1))[0]
        hist_glom = np.histogram(mf_glom[:, 1], bins=np.append(glom, np.max(glom)+1))[0]
        self.assertEqual(np.mean(hist_glom), 1.0)
        self.assertTrue(19 <= np.mean(hist_mf) <= 21)


    def test_MF_DCN(self):
        mossy = self.nest_adapter.entities['mossy_fibers'].nest_identifiers
        dcn = self.nest_adapter.cell_models['dcn_cell'].nest_identifiers
        mf_dcn = np.array(self.nest_adapter.nest.GetConnections(mossy, dcn))
        hist_mf = np.histogram(mf_dcn[:, 0], bins=np.append(mossy, np.max(mossy)+1))[0]
        hist_dcn = np.histogram(mf_dcn[:, 1], bins=np.append(dcn, np.max(dcn)+1))[0]
        self.assertEqual(np.mean(hist_dcn), 50)
        self.assertTrue(1 <= np.mean(hist_mf) <= 4)
