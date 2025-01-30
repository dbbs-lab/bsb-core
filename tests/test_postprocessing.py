import os
import unittest

import numpy as np
from bsb_test import NumpyTestCase, RandomStorageFixture

from bsb import (
    MPI,
    AfterConnectivityHook,
    AfterPlacementHook,
    Configuration,
    FuseConnections,
    Scaffold,
    config,
)


class TestAfterConnectivityHook(
    RandomStorageFixture, unittest.TestCase, engine_name="hdf5"
):
    def setUp(self):
        super().setUp()

        @config.node
        class TestAfterConn(AfterConnectivityHook):
            def postprocess(self):
                with open(f"test_after_conn_{MPI.get_rank()}.txt", "a") as f:
                    # make sure we have access to the scaffold context
                    f.write(f"{self.scaffold.configuration.name}\n")

        self.network = Scaffold(
            config=Configuration.default(
                name="Test config",
                after_connectivity={"test_after_conn": TestAfterConn()},
            ),
            storage=self.storage,
        )

    def test_after_connectivity_job(self):
        self.network.compile()
        if MPI.get_rank() == 0:
            count_files = 0
            for filename in os.listdir():
                if filename.startswith("test_after_conn_"):
                    count_files += 1
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        self.assertEqual(
                            len(lines), 1, "The postprocess should be called only once."
                        )
                        self.assertEqual(lines[0], "Test config\n")
                    os.remove(filename)
            self.assertEqual(count_files, 1)


class TestAfterPlacementHook(RandomStorageFixture, unittest.TestCase, engine_name="hdf5"):
    def setUp(self):
        super().setUp()

        @config.node
        class TestAfterPlace(AfterPlacementHook):
            def postprocess(self):
                with open(f"test_after_place_{MPI.get_rank()}.txt", "a") as f:
                    # make sure we have access to the scaffold context
                    f.write(f"{self.scaffold.configuration.name}\n")

        self.network = Scaffold(
            config=Configuration.default(
                name="Test config",
                after_placement={"test_after_placement": TestAfterPlace()},
            ),
            storage=self.storage,
        )

    def test_after_placement_job(self):
        self.network.compile()
        if MPI.get_rank() == 0:
            count_files = 0
            for filename in os.listdir():
                if filename.startswith("test_after_place_"):
                    count_files += 1
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        self.assertEqual(
                            len(lines), 1, "The postprocess should be called only once."
                        )
                        self.assertEqual(lines[0], "Test config\n")
                    os.remove(filename)
            self.assertEqual(count_files, 1)


class TestFuseConnectionsHook(
    RandomStorageFixture,
    NumpyTestCase,
    unittest.TestCase,
    engine_name="hdf5",
):
    def setUp(self):
        super().setUp()
        self.cfg = Configuration.default(
            network={"x": 100.0, "y": 100.0, "z": 100.0, "chunk_size": [100, 100, 50]},
            regions={
                "brain_region": {"type": "stack", "children": ["base_layer", "top_layer"]}
            },
            partitions=dict(
                base_layer=dict(type="layer", thickness=50),
                top_layer=dict(type="layer", thickness=50),
            ),
            cell_types=dict(
                A=dict(spatial=dict(radius=1, count=2)),
                B=dict(spatial=dict(radius=1, count=4)),
                C=dict(spatial=dict(radius=1, count=2)),
                D=dict(spatial=dict(radius=1, count=3)),
            ),
            placement=dict(
                top_placement={
                    "strategy": "bsb.placement.RandomPlacement",
                    "cell_types": ["B", "D"],
                    "partitions": ["top_layer"],
                },
                base_placement={
                    "strategy": "bsb.placement.RandomPlacement",
                    "cell_types": ["A"],
                    "partitions": ["base_layer"],
                },
                other_placement={
                    "strategy": "bsb.placement.RandomPlacement",
                    "cell_types": ["C"],
                    "partitions": ["base_layer"],
                },
            ),
        )
        self.network = Scaffold(self.cfg, self.storage)
        self.network.compile(skip_connectivity=True)

        # Set custom connections only on master rank
        a_to_b = -1 * np.ones((2, 4, 3))
        a_to_b[0] = [[0, 1, 1], [1, 1, 1], [1, 2, 1], [1, 2, 2]]
        a_to_b[1, :, 0] = [0, 0, 1, 2]

        b_to_c = -1 * np.ones((2, 4, 3))
        b_to_c[0, :, 0] = [0, 0, 1, 3]
        b_to_c[1, :, 0] = [0, 1, 0, 1]

        c_to_d = -1 * np.ones((2, 5, 3))
        c_to_d[0, :, 0] = [0, 0, 1, 1, 1]
        c_to_d[1] = [[0, 2, 1], [2, -1, -1], [0, 1, 1], [1, -1, -1], [2, 1, 2]]

        b_to_d = -1 * np.ones((2, 2, 3))
        b_to_d[0, :, 0] = [0, 3]
        b_to_d[1, :, 0] = [1, 0]

        d_to_a = -1 * np.ones((2, 1, 3))
        d_to_a[0, :, 0] = [0]
        d_to_a[1, :, 0] = [0]

        self.a_to_b = a_to_b
        self.b_to_c = b_to_c
        self.c_to_d = c_to_d
        if not MPI.get_rank():

            ps_a = self.network.cell_types["A"].get_placement_set()
            ps_b = self.network.cell_types["B"].get_placement_set()
            ps_c = self.network.cell_types["C"].get_placement_set()
            ps_d = self.network.cell_types["D"].get_placement_set()

            self.network.connect_cells(ps_a, ps_b, a_to_b[0], a_to_b[1], "A_to_B")
            self.network.connect_cells(ps_b, ps_c, b_to_c[0], b_to_c[1], "B_to_C")
            self.network.connect_cells(ps_c, ps_d, c_to_d[0], c_to_d[1], "C_to_D")
            self.network.connect_cells(ps_b, ps_d, b_to_d[0], b_to_d[1], "B_to_D")
            self.network.connect_cells(ps_d, ps_a, d_to_a[0], d_to_a[1], "D_to_A")

    def test_nonexistent_set(self):

        self.cfg.after_connectivity = dict(
            new_connection=dict(
                strategy="bsb.postprocessing.FuseConnections",
                connections=["B_to_C", "K_to_B"],
            )
        )

        with self.assertRaises(Exception) as e:
            self.network.run_after_connectivity()

    def test_merge_sets(self):

        my_hook = FuseConnections(connections=["A_to_B", "B_to_C"])

        computed_connections = my_hook.merge_sets(self.a_to_b, self.b_to_c)
        real_connections = (
            np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 2, 1]]),
            np.array([[0, -1, -1], [1, -1, -1], [0, -1, -1], [1, -1, -1], [0, -1, -1]]),
        )

        self.assertAll(
            computed_connections[0] == real_connections[0],
            "Fused connection must match real connections",
        )
        self.assertAll(
            computed_connections[1] == real_connections[1],
            "Fused connection must match real connections",
        )

    def test_no_branches(self):
        # Test that the branch B + C -> D is detected

        self.cfg.after_connectivity = dict(
            new_connection=dict(
                strategy="bsb.postprocessing.FuseConnections",
                connections=["A_to_B", "B_to_C", "C_to_D", "B_to_D"],
            )
        )

        with self.assertRaises(Exception) as e:
            self.network.run_after_connectivity()

    def test_no_loops(self):
        # Test that a loop is detected

        self.cfg.after_connectivity = dict(
            new_connection=dict(
                strategy="bsb.postprocessing.FuseConnections",
                connections=["A_to_B", "B_to_C", "D_to_A", "C_to_D"],
            )
        )
        with self.assertRaises(Exception) as e:
            self.network.run_after_connectivity()

    def test_three_connectivities(self):
        # Test the chained A_B -> B_C -> C_D fusion.

        self.cfg.after_connectivity = dict(
            new_connection=dict(
                strategy="bsb.postprocessing.FuseConnections",
                connections=["B_to_C", "A_to_B", "C_to_D"],
            )
        )
        self.network.run_after_connectivity()

        A_locs = [[0, 1, 1], [1, 1, 1], [1, 2, 1], [1, 2, 2]]
        D_locs = self.c_to_d[1]

        real_connections = (
            (np.repeat(A_locs[0:3:], [5, 5, 2], axis=0)),
            np.append(np.concatenate((D_locs, D_locs), axis=0), D_locs[0:2], axis=0),
        )
        cs = self.network.get_connectivity_set("new_connection")
        computed_connections = cs.load_connections().all()

        self.assertEqual(
            len(computed_connections[0]),
            len(computed_connections[1]),
            "Lenghts of pre_locs and post_locs connections do not match!",
        )
        self.assertEqual(
            len(computed_connections[0]),
            len(real_connections[0]),
            "Number of computed connections do not match the number of real connections",
        )

        # Create a transponse of the arrays and check if we obtain the same connections

        reversed_comp = np.array(
            [
                (ele[0], ele[1])
                for ele in zip(computed_connections[0], computed_connections[1])
            ]
        )
        reversed_real = np.array(
            [(ele[0], ele[1]) for ele in zip(real_connections[0], real_connections[1])]
        )

        check_all = np.zeros(len(reversed_real))
        for i, real in enumerate(reversed_real):
            for comp in reversed_comp:
                check_all[i] += np.all(real == comp)

        self.assertAll(check_all == 1, f"Some fused connections do not match real ones!")
