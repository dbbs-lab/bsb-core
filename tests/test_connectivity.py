import unittest, os, sys, numpy as np, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.core import Scaffold
from scaffold.models import Layer, CellType
from test_setup import get_test_network


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


class TestConnectivity(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestConnectivity, self).setUpClass()
        self.scaffold = get_test_network(200, 200)
        self.nest_adapter = self.scaffold.get_simulation("FCN_2019")
        self.scaffold.run_simulation("FCN_2019")

    def test_MF_Glom(self):
        mossy = self.nest_adapter.entities["mossy_fibers"].nest_identifiers
        glom = self.nest_adapter.cell_models["glomerulus"].nest_identifiers
        mf_glom = np.array(self.nest_adapter.nest.GetConnections(mossy, glom))
        hist_mf = np.histogram(mf_glom[:, 0], bins=np.append(mossy, np.max(mossy) + 1))[0]
        hist_glom = np.histogram(mf_glom[:, 1], bins=np.append(glom, np.max(glom) + 1))[0]
        self.assertEqual(np.mean(hist_glom), 1.0)
        self.assertTrue(19 <= np.mean(hist_mf) <= 21)

    def test_MF_DCN(self):
        mossy = self.nest_adapter.entities["mossy_fibers"].nest_identifiers
        dcn = self.nest_adapter.cell_models["dcn_cell"].nest_identifiers
        mf_dcn = np.array(self.nest_adapter.nest.GetConnections(mossy, dcn))
        hist_mf = np.histogram(mf_dcn[:, 0], bins=np.append(mossy, np.max(mossy) + 1))[0]
        hist_dcn = np.histogram(mf_dcn[:, 1], bins=np.append(dcn, np.max(dcn) + 1))[0]
        self.assertEqual(np.mean(hist_dcn), 50)
        self.assertTrue(1 <= np.mean(hist_mf) <= 4)

    def test_Glom_GoC(self):
        glom = self.nest_adapter.cell_models["glomerulus"].nest_identifiers
        goc = self.nest_adapter.cell_models["golgi_cell"].nest_identifiers
        glom_goc = self.nest_adapter.nest.GetConnections(glom, goc)
        self.assertTrue(glom_goc)

    def test_GoC_GoC(self):
        goc = self.nest_adapter.cell_models["golgi_cell"].nest_identifiers
        goc_goc = self.nest_adapter.nest.GetConnections(goc, goc)
        self.assertTrue(goc_goc)

    # Test goc_glom connectivity in the scaffold creation (not used in NEST)
    def test_GoC_Glom(self):
        first_goc_id = np.int(min(self.scaffold.get_cells_by_type("golgi_cell")[:, 0]))
        last_goc_id = np.int(max(self.scaffold.get_cells_by_type("golgi_cell")[:, 0]))
        first_glom_id = np.int(min(self.scaffold.get_cells_by_type("glomerulus")[:, 0]))
        last_glom_id = np.int(max(self.scaffold.get_cells_by_type("glomerulus")[:, 0]))
        goc_glom = self.scaffold.get_connections_by_cell_type(
            presynaptic="golgi_cell", postsynaptic="glomerulus"
        )
        self.assertTrue(
            first_goc_id <= min(goc_glom[0][1].from_identifiers) <= last_goc_id
        )
        self.assertTrue(
            first_glom_id <= min(goc_glom[0][1].to_identifiers) <= last_glom_id
        )
