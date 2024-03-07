import os
import unittest

from bsb import config
from bsb.cell_types import CellType
from bsb.config.refs import Reference
from bsb.exceptions import CfgReferenceError
from bsb.placement import FixedPositions
from bsb.placement.indicator import PlacementIndications
from bsb.storage import Chunk


def relative_to_tests_folder(path):
    return os.path.join(os.path.dirname(__file__), path)


class BothReference(Reference):
    def __call__(self, root, here):
        merged = root.examples.copy()
        merged.update(root.extensions)
        return merged

    def is_ref(self, value):
        return not isinstance(value, str)


@config.node
class Example:
    mut_ex = config.attr(required=True)


@config.node
class Extension:
    ex_mut = config.attr(type=int, required=True)
    ref = config.ref(BothReference(), required=True)


@config.root
class Root430:
    examples = config.dict(type=Example, required=True)
    extensions = config.dict(type=Extension, required=True)


class TestIssues(unittest.TestCase):
    def test_430(self):
        with self.assertRaises(CfgReferenceError, msg="Regression of issue #430"):
            config = Root430(
                examples=dict(), extensions=dict(x=dict(ex_mut=4, ref="missing"))
            )
            print("ref", config.extensions.x.ref)

    def test_802(self):
        """
        Test fixed positions with 0 positions
        """
        with self.assertWarns(UserWarning):
            # Cobble some components together to test `positions=[]`
            FixedPositions(positions=[], cell_types=[], partitions=[]).place(
                Chunk((0, 0, 0), (100, 100, 100)),
                {CellType(spatial=dict(radius=1, count=1)): PlacementIndications()},
            )
