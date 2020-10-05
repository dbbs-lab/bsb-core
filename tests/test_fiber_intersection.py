import unittest, os, sys, numpy as np, h5py, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bsb.core import Scaffold
from bsb.config import JSONConfig
from bsb.models import Layer, CellType, ConnectivitySet
from bsb.output import MorphologyRepository
import test_setup


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


fiber_transform_config = relative_to_tests_folder("configs/test_fiber_intersection.json")
morpho_file = relative_to_tests_folder("morphologies.hdf5")


class TestFiberIntersection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestFiberIntersection, self).setUpClass()
        # Make sure the MR exists
        test_setup.prep_morphologies()
        # The scaffold has only the Granular layer (100x100x150) with 20 GrCs
        # and 1 GoC placed, as specified in the config file
        self.config = JSONConfig(file=fiber_transform_config)
        # Defining quivers field to include also voxels outside the scaffold
        # volume
        self.quivers_field = np.zeros(
            shape=(3, 80, 80, 80)
        )  # Each voxel in the volume has vol_res=25um
        # Fake quiver, oriented as original fibers
        basic_quiver = np.array([0, 1.0, 0.0])
        self.quivers_field[0, :] = basic_quiver[0]
        self.quivers_field[1, :] = basic_quiver[1]
        self.quivers_field[2, :] = basic_quiver[2]
        self.config.connection_types[
            "parallel_fiber_to_golgi_bended"
        ].transformation.quivers = self.quivers_field
        self.config.connection_types[
            "parallel_fiber_to_golgi_bended"
        ].transformation.vol_start = [-500.0, -500.0, -500.0]
        self.scaffold = Scaffold(self.config)
        self.scaffold.morphology_repository = MorphologyRepository(morpho_file)
        self.scaffold.compile_network()

    def test_fiber_connections(self):
        pre_type = "granule_cell"
        pre_neu = self.scaffold.get_placement_set(pre_type)
        conn_type = "parallel_fiber_to_golgi"
        cs = self.scaffold.get_connectivity_set(conn_type)
        num_conn = len(cs.connections)
        # Check that no more connections are formed than the number of
        # presynaptic neurons - how could happen otherwise?
        self.assertTrue(num_conn <= len(pre_neu.identifiers))

        # Check that increasing resolution in FiberIntersection does not change
        # connection number if there are no transformations (and thus the fibers
        # are parallel to main axes)
        conn_type_HR = "parallel_fiber_to_golgi_HR"
        cs_HR = self.scaffold.get_connectivity_set(conn_type_HR)
        self.assertEqual(num_conn, len(cs_HR.connections))

        # Check that half (+- 5) connections are obtained with half the affinity
        conn_type_affinity = "parallel_fiber_to_golgi_affinity"
        cs_affinity = self.scaffold.get_connectivity_set(conn_type_affinity)
        self.assertAlmostEqual(num_conn / 2, len(cs_affinity.connections), delta=5)

        # Check that same number of connections are obtained when a fake quiver
        # transformation is applied
        conn_type_transform = "parallel_fiber_to_golgi_bended"
        cs_fake_transform = self.scaffold.get_connectivity_set(conn_type_transform)
        self.assertEqual(len(cs_fake_transform.connections), num_conn)

        # Check that less connections are obtained when the PC surface is
        # oriented according to orientation vector of 45° rotation in yz plane,
        # for how the Golgi cell is placed and the parallel fibers are rotated
        basic_quiver = np.array([0, 0.7, 0.7])
        self.quivers_field[0, :] = basic_quiver[0]
        self.quivers_field[1, :] = basic_quiver[1]
        self.quivers_field[2, :] = basic_quiver[2]
        self.config.connection_types[
            "parallel_fiber_to_golgi_bended"
        ].transformation.quivers = self.quivers_field
        self.scaffold = Scaffold(self.config)
        self.scaffold.compile_network()
        cs_transform = self.scaffold.get_connectivity_set(conn_type_transform)
        self.assertTrue(len(cs_transform.connections) <= num_conn)


class TestBranching(unittest.TestCase):
    pass
