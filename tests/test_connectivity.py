import unittest, os, sys, numpy as np, h5py, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.core import Scaffold
from scaffold.models import Layer, CellType
from test_setup import get_test_network


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


_nest_available = importlib.util.find_spec("nest") is not None
_using_morphologies = True


@unittest.skipIf(_using_morphologies, "Morphologies are used for the connectivity")
class TestConnectivity(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestConnectivity, self).setUpClass()
        self.scaffold = get_test_network(200, 200)
        if _nest_available:
            self.nest_adapter = self.scaffold.get_simulation("FCN_2019")
            self.scaffold.run_simulation("FCN_2019")

    @unittest.skipIf(not _nest_available, "NEST is not importable.")
    def test_MF_Glom(self):
        mossy = self.nest_adapter.entities["mossy_fibers"].nest_identifiers
        glom = self.nest_adapter.cell_models["glomerulus"].nest_identifiers
        mf_glom = np.array(self.nest_adapter.nest.GetConnections(mossy, glom))
        hist_mf = np.histogram(mf_glom[:, 0], bins=np.append(mossy, np.max(mossy) + 1))[0]
        hist_glom = np.histogram(mf_glom[:, 1], bins=np.append(glom, np.max(glom) + 1))[0]
        self.assertEqual(np.mean(hist_glom), 1.0)
        self.assertTrue(19 <= np.mean(hist_mf) <= 21)

    @unittest.skipIf(not _nest_available, "NEST is not importable.")
    def test_MF_DCN(self):
        mossy = self.nest_adapter.entities["mossy_fibers"].nest_identifiers
        dcn = self.nest_adapter.cell_models["dcn_cell"].nest_identifiers
        mf_dcn = np.array(self.nest_adapter.nest.GetConnections(mossy, dcn))
        hist_mf = np.histogram(mf_dcn[:, 0], bins=np.append(mossy, np.max(mossy) + 1))[0]
        hist_dcn = np.histogram(mf_dcn[:, 1], bins=np.append(dcn, np.max(dcn) + 1))[0]
        self.assertEqual(np.mean(hist_dcn), 50)
        self.assertTrue(1 <= np.mean(hist_mf) <= 4)

    @unittest.skipIf(not _nest_available, "NEST is not importable.")
    def test_Glom_GoC(self):
        glom = self.nest_adapter.cell_models["glomerulus"].nest_identifiers
        goc = self.nest_adapter.cell_models["golgi_cell"].nest_identifiers
        glom_goc = self.nest_adapter.nest.GetConnections(glom, goc)
        self.assertTrue(glom_goc)

    @unittest.skipIf(not _nest_available, "NEST is not importable.")
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

        # Tests if labelled cells are connected only with cells with the same label

    def test_Microzones(self):

        micro_neg = self.scaffold.get_labelled_ids("microzone-negative")
        micro_pos = self.scaffold.get_labelled_ids("microzone-positive")
        for pre in ["purkinje_cell", "dcn_cell", "dcn_interneuron", "io_cell"]:
            for post in ["purkinje_cell", "dcn_cell", "dcn_interneuron", "io_cell"]:
                pre_post = self.scaffold.get_connections_by_cell_type(
                    presynaptic=pre, postsynaptic=post
                )
                if len(pre_post) > 0:
                    for conn_type in pre_post[0][1:]:
                        A_to_B = np.column_stack(
                            (conn_type.from_identifiers, conn_type.to_identifiers)
                        )
                        for connection_i in A_to_B:
                            self.assertTrue(
                                (connection_i[0] in micro_neg)
                                == (connection_i[1] in micro_neg)
                            )

    def test_connectivity_matrix(self):

        for connections in self.scaffold.configuration.connection_types.values():
            for connection_tag in connections.tags:
                cs = self.scaffold.get_connectivity_set(connection_tag)
                from_cells = cs.from_identifiers
                to_cells = cs.to_identifiers
                connections = cs.get_dataset()
                pre = np.unique(connections[:, 0])
                post = np.unique(connections[:, 1])

                with self.subTest(name="PRE " + connection_tag):
                    for conn in range(len(pre)):
                        self.assertTrue(pre[conn] in from_cells)

                with self.subTest(name="POST " + connection_tag):
                    for conn in range(len(post)):
                        self.assertTrue(post[conn] in to_cells)

                # Call convergence and divergence code.
                with self.subTest(name="divergence"):
                    _ = cs.divergence
                with self.subTest(name="convergence"):
                    _ = cs.convergence
