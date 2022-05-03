import unittest, os, sys, numpy as np, h5py, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bsb.core import Scaffold
from bsb import config
from bsb.config import from_json, Configuration
from bsb.config import types
from bsb.exceptions import *
from test_setup import get_config
from bsb.topology import Region, Partition


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


@config.root
class Root430:
    regions = config.dict(type=Region)
    partitions = config.dict(type=Partition, required=True)


class TestIssues(unittest.TestCase):
    def test_430(self):
        with self.assertRaises(ReferenceError, msg="Regression of issue #430"):
            config = Root430(
                regions=dict(), partitions=dict(x=dict(region="missing", thickness=10))
            )
