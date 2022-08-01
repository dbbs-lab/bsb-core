import unittest, os, sys, numpy as np, h5py, json, string, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.exceptions import *
from bsb.core import Scaffold
from bsb.cell_types import CellType
from bsb.config import from_json, Configuration
from bsb.morphologies.selector import NameSelector
from bsb.placement.distributor import MorphologyDistributor
from bsb.unittest import skip_parallel


class TestMorphologyDistributor(unittest.TestCase):
    @skip_parallel
    # Errors during parallel jobs cause MPI_Abort, untestable scenario.
    def test_empty_selection(self):
        cfg = Configuration.default(
            regions=dict(reg=dict(children=["a"])),
            partitions=dict(a=dict(thickness=100)),
            cell_types=dict(
                a=dict(spatial=dict(radius=2, density=1e-3, morphologies=[{"names": []}]))
            ),
            placement=dict(
                a=dict(
                    strategy="bsb.placement.RandomPlacement",
                    cell_types=["a"],
                    partitions=["a"],
                )
            ),
        )
        netw = Scaffold(cfg)
        with self.assertRaisesRegex(DistributorError, "NameSelector"):
            netw.compile(append=True)
