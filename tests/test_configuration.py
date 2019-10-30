import unittest, os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scaffold import Scaffold
from scaffold.config import JSONConfig
from scaffold.models import Layer, CellType

class TestConfiguration(unittest.TestCase):

    def test_minimal(self):
        config = JSONConfig(file="configs/test_minimal.json")
        self.scaffold = Scaffold(config)
