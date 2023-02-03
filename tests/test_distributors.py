import unittest
import numpy as np
from bsb.exceptions import DistributorError, DatasetNotFoundError, EmptySelectionError
from bsb.core import Scaffold
from bsb.config import Configuration
from bsb.unittest import skip_parallel, get_data_path
from bsb.placement.distributor import (
    MorphologyDistributor,
    MorphologyGenerator,
    VolumetricRotations,
)
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
                a=dict(spatial=dict(radius=2, density=1e-4, morphologies=[{"names": []}]))
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

    def test_same_generators(self):
        self.netw.placement.a.distribute.morphologies = SameEmptyGenerator()
        self.netw.compile()
        ps = self.netw.get_placement_set("a")
        ms = ps.load_morphologies()
        morphologies = list(ms.iter_morphologies(unique=True))
        self.assertEqual(len(ps), len(ms), "equal data")
        self.assertEqual(
            len(ps.get_loaded_chunks()),
            len(morphologies),
            "expected each chunk to generate 1 unique empty morphology",
        )

    def test_many_generators(self):
        self.netw.placement.a.distribute.morphologies = ManyEmptyGenerator()
        self.netw.compile()
        ps = self.netw.get_placement_set("a")
        ms = ps.load_morphologies()
        morphologies = list(ms.iter_morphologies(unique=True))
        self.assertEqual(len(ps), len(ms), "equal data")
        self.assertEqual(
            len(ps), len(morphologies), "expected 1 unique morphology per cell"
        )


class TestVolumetricRotations(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration.default(
            regions=dict(reg=dict(children=["a"])),
            partitions=dict(
                a=dict(
                    type="nrrd",
                    source=get_data_path("orientations", "toy_annotations.nrrd"),
                    voxel_size=25,
                )
            ),
            cell_types=dict(a=dict(spatial=dict(radius=2, density=1e-4))),
            placement=dict(
                a=dict(
                    strategy="bsb.placement.RandomPlacement",
                    cell_types=["a"],
                    partitions=["a"],
                    distribute=dict(
                        rotations=VolumetricRotations(
                            orientation_path=get_data_path(
                                "orientations", "toy_orientations.nrrd"
                            ),
                            orientation_resolution=25.0,
                            default_vector=np.array([-1.0, 0.0, 0.0]),
                        ),
                    ),
                ),
            ),
        )
        self.netw = Scaffold(self.cfg)

    def test_distribute(self):
        self.netw.compile(clear=True)
        positions = self.netw.get_placement_set("a").load_positions()
        voxel_set = self.netw.partitions.a.get_voxelset()
        region_ids = np.asarray(
            voxel_set.data[:, 0][voxel_set.index_of(positions)], dtype=int
        )
        rotations = np.array(self.netw.get_placement_set("a").load_rotations())
        # Regions without orientation field -> no rotation
        self.assertTrue(
            np.array_equal(
                np.all(rotations == 0.0, axis=1), np.isin(region_ids, (100, 728, 744))
            )
        )
        # Regions with orientation field -> AIBS Flocculus and Lingula
        pos_w_rot = np.any(rotations != 0, axis=1)
        self.assertTrue(
            np.array_equal(
                pos_w_rot, np.isin(region_ids, (10690, 10691, 10692, 10705, 10706, 10707))
            )
        )
        self.assertTrue(
            np.all((-180.0 < rotations[pos_w_rot]) * (rotations[pos_w_rot] < 180.0))
        )
        # orientation field x component should be close to 0.
        self.assertTrue(np.all(np.absolute(rotations[pos_w_rot][:, 0]) < 0.5))
