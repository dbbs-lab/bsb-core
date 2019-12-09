import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold.scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.models import Layer, CellType


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


minimal_config = relative_to_tests_folder("configs/test_minimal.json")


class TestConfiguration(unittest.TestCase):

    def test_minimal(self):
        config = JSONConfig(file=minimal_config)
        self.scaffold = Scaffold(config)
