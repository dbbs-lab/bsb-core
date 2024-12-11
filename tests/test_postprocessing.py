import os
import unittest

from bsb_test import RandomStorageFixture

from bsb import (
    MPI,
    AfterConnectivityHook,
    AfterPlacementHook,
    Configuration,
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
        count_files = 0
        for filename in os.listdir():
            if filename.startswith(f"test_after_conn_{MPI.get_rank()}"):
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
        count_files = 0
        for filename in os.listdir():
            if filename.startswith(f"test_after_place_{MPI.get_rank()}"):
                count_files += 1
                with open(filename, "r") as f:
                    lines = f.readlines()
                    self.assertEqual(
                        len(lines), 1, "The postprocess should be called only once."
                    )
                    self.assertEqual(lines[0], "Test config\n")
                os.remove(filename)
        self.assertEqual(count_files, 1)
