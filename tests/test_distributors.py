import unittest
from bsb.exceptions import DistributorError, DatasetNotFoundError, EmptySelectionError
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb.unittest import skip_parallel
from bsb.placement.distributor import MorphologyDistributor, MorphologyGenerator
from bsb.morphologies import Morphology


class OneNoneDistributor(MorphologyDistributor):
    def distribute(self, *args):
        return None


class TupleNoneDistributor(MorphologyDistributor):
    def distribute(self, *args):
        return None, None


class SameEmptyGenerator(MorphologyGenerator):
    def generate(self, pos, loaders, context):
        return [Morphology.empty()] * len(pos)


class ManyEmptyGenerator(MorphologyGenerator):
    def generate(self, pos, loaders, context):
        return [Morphology.empty() for _ in range(len(pos))]


class TestMorphologyDistributor(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration.default(
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
        self.netw = Scaffold(self.cfg)

    @skip_parallel
    # Errors during parallel jobs cause MPI_Abort, untestable scenario.
    def test_empty_selection(self):
        with self.assertRaisesRegex(DistributorError, "NameSelector"):
            self.netw.compile(append=True)

    @skip_parallel
    # Errors during parallel jobs cause MPI_Abort, untestable scenario.
    def test_none_returns(self):
        self.netw.morphologies.save("bs", Morphology.empty(), overwrite=True)
        self.netw.cell_types.a.spatial.morphologies = [{"names": ["*"]}]
        self.netw.placement.a.distribute.morphologies = OneNoneDistributor()
        self.netw.compile(append=True)
        ps = self.netw.get_placement_set("a")
        self.assertTrue(len(ps) > 0, "should've still placed cells")
        with self.assertRaises(DatasetNotFoundError, msg="shouldnt have morphos"):
            ps.load_morphologies()
        self.netw.placement.a.distribute.morphologies = TupleNoneDistributor()
        self.netw.compile(append=True)
        self.assertTrue(len(ps) > 0, "should've still placed cells")
        with self.assertRaises(DatasetNotFoundError, msg="shouldnt have morphos"):
            ps.load_morphologies()

    def test_generators(self):
        self.netw.placement.a.distribute.morphologies = SameEmptyGenerator()
        self.netw.compile()
