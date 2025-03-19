import unittest
from time import sleep

import numpy as np
from bsb_test import (
    NetworkFixture,
    NumpyTestCase,
    RandomStorageFixture,
    get_test_config,
    skip_parallel,
)

from bsb import (
    MPI,
    BootError,
    CellType,
    Chunk,
    Configuration,
    IndicatorError,
    PackingWarning,
    Partition,
    PlacementError,
    PlacementRelationError,
    PlacementStrategy,
    Rhomboid,
    Scaffold,
    VoxelData,
    Voxels,
    VoxelSet,
    WorkflowError,
)


def dud_tester(scaffold, x, y):
    sleep(y)
    return x


def chunk_tester(scaffold, chunk):
    return chunk


class PlacementDud(PlacementStrategy):
    name = "dud"

    def place(self, chunk, indicators):
        pass


class VoxTest(Voxels, classmap_entry="vox_test"):
    loc_data = np.random.rand(8, 8, 8)

    def get_voxelset(self):
        return VoxelSet(
            np.transpose(np.nonzero(self.loc_data)),
            np.array([100, 100, 100], dtype=int),
            data=self.loc_data.flatten(),
            data_keys=["vox_density"],
        )


def single_layer_placement(network, offset=None):
    # fixme: https://github.com/dbbs-lab/bsb-core/issues/812
    network.topology.children.append(part := Partition(name="dud_layer", thickness=120))
    network.network.origin = offset if offset is not None else [0.0, 0.0, 0.0]
    network.resize()
    dud_cell = CellType(name="cell_w_count", spatial={"count": 40, "radius": 2})
    dud_cell2 = CellType(
        name="cell_rel_count",
        spatial={"relative_to": "cell_w_count", "count_ratio": 0.5, "radius": 3.4},
    )
    dud_cell7 = CellType(name="cell_no_ind", spatial={"radius": 1.0})
    network.cell_types["cell_w_count"] = dud_cell
    network.cell_types["cell_rel_count"] = dud_cell2
    network.cell_types["cell_no_ind"] = dud_cell7
    dud = PlacementDud(
        name="dud",
        strategy="PlacementDud",
        partitions=[part],
        cell_types=[dud_cell, dud_cell2, dud_cell7],
        overrides={"cell_w_count": {}},
    )
    network.placement["dud"] = dud
    return dud


def single_vox_placement(network, voxels):
    network.topology.children.append(voxels)
    network.network.origin = [0.0, 0.0, 0.0]
    network.resize()
    dud_cell = CellType(
        name="cell_density_key",
        spatial={"density_key": "vox_density", "radius": 5.6},
    )
    dud_cell2 = CellType(
        name="cell_rel_dens_key",
        spatial={"relative_to": "cell_density_key", "density_ratio": 2.0, "radius": 5.6},
    )
    dud_cell3 = CellType(
        name="cell_rel_dens_no_ratio",
        spatial={"relative_to": "cell_density_key", "radius": 5.6},
    )
    dud_cell4 = CellType(
        name="cell_w_dens",
        spatial={"density": 0.1, "radius": 5.6},
    )
    network.cell_types["cell_density_key"] = dud_cell
    network.cell_types["cell_rel_dens_key"] = dud_cell2
    network.cell_types["cell_rel_dens_no_ratio"] = dud_cell3
    network.cell_types["cell_w_dens"] = dud_cell4
    dud2 = PlacementDud(
        name="dud2",
        strategy="PlacementDud",
        partitions=[voxels],
        cell_types=[dud_cell2, dud_cell, dud_cell3, dud_cell4],
        overrides={"cell_rel_dens_key": {}, "cell_density_key": {}},
    )
    network.placement["dud2"] = dud2
    return dud2


def _chunk(x, y, z):
    return Chunk((x, y, z), (100, 100, 100))


class TestIndicators(
    RandomStorageFixture, NetworkFixture, unittest.TestCase, engine_name="fs"
):
    def setUp(self):
        self.cfg = Configuration.default()
        super().setUp()
        self.voxels = VoxTest()
        self.placement = single_layer_placement(self.network)
        self.placement2 = single_vox_placement(self.network, self.voxels)

    def test_cascade(self):
        indicators = self.placement.get_indicators()
        dud_ind = indicators["cell_w_count"]
        dud2_ind = indicators["cell_rel_count"]
        self.assertEqual(2, dud_ind.indication("radius"))
        self.assertEqual(40, dud_ind.indication("count"))
        self.assertEqual(2, dud_ind.get_radius())
        self.placement.overrides.cell_w_count.radius = 4
        self.assertEqual(4, dud_ind.indication("radius"))
        self.placement.overrides.cell_w_count.radius = None
        self.placement.cell_types[0].spatial.radius = None
        self.assertEqual(None, dud_ind.indication("radius"))
        self.assertRaises(IndicatorError, dud_ind.get_radius)
        self.assertTrue(dud2_ind.indication("relative_to") in self.placement.cell_types)
        self.assertEqual(dud2_ind.indication("count_ratio"), 0.5)
        indicators = self.placement2.get_indicators()
        dud3_ind = indicators["cell_rel_dens_key"]
        self.assertEqual(dud3_ind.indication("density_ratio"), 2.0)

    def test_guess_count(self):
        indicators = self.placement.get_indicators()
        dud_ind = indicators["cell_w_count"]
        dud2_ind = indicators["cell_rel_count"]
        ratio_dud2 = 0.5
        self.assertEqual(40, dud_ind.guess())
        self.assertEqual(40 * ratio_dud2, dud2_ind.guess())
        self.placement.overrides.cell_w_count.count = 400
        self.assertEqual(400, dud_ind.guess())
        self.assertEqual(400 * ratio_dud2, dud2_ind.guess())
        bottom_ratio = 1 / 1.2
        bottom = 400 * bottom_ratio / 4
        top_ratio = 0.2 / 1.2
        top = 400 * top_ratio / 4
        for x, y, z in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(bottom) <= guess <= np.ceil(bottom))
        for x, y, z in ((0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(top) <= guess <= np.ceil(top))
        for x, y, z in ((0, 0, -1), (0, 0, 2), (2, 0, 1), (1, -3, 1)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertEqual(0, guess)
        with self.assertRaises(IndicatorError):
            indicators["cell_no_ind"].guess()

    def test_guess_vox_density(self):
        indicators = self.placement2.get_indicators()
        dud3_ind = indicators["cell_rel_dens_key"]
        dud4_ind = indicators["cell_density_key"]
        ratio_dud3 = 2.0
        predicted_count = (self.voxels.loc_data * 100**3).flatten()
        guess4 = dud4_ind.guess()
        self.assertTrue(abs(np.sum(predicted_count) - guess4) <= 1)

        guess4 = dud4_ind.guess(voxels=self.voxels.get_voxelset())
        guess3 = dud3_ind.guess(voxels=self.voxels.get_voxelset())
        self.assertTrue(np.all(np.absolute(predicted_count - guess4) <= 1))

        self.assertTrue(
            np.all(
                np.absolute((predicted_count * ratio_dud3) - guess3)
                <= np.ceil(ratio_dud3)
            )
        )
        self.placement2.overrides.cell_rel_dens_key.relative_to = self.network.cell_types[
            "cell_w_count"
        ]
        with self.assertRaises(PlacementRelationError):
            # Cannot estimate relative to an estimate without density key
            dud3_ind.guess()
        with self.assertRaises(PlacementError):
            # Cannot estimate relative to without a ratio
            indicators["cell_rel_dens_no_ratio"].guess()

        self.assertEqual(
            0.1 * self.voxels.loc_data.size * 100**3, indicators["cell_w_dens"].guess()
        )
        self.placement2.overrides.cell_rel_dens_key.relative_to = self.network.cell_types[
            "cell_w_dens"
        ]
        self.assertEqual(0.2 * self.voxels.loc_data.size * 100**3, dud3_ind.guess())

        self.placement2.overrides.cell_density_key.density_key = "bla"
        with self.assertRaises(RuntimeError):
            # voxel density key not found
            dud4_ind.guess(voxels=self.voxels.get_voxelset())

    def test_regression_issue_885(self):
        # Test placement with count ratio in separated partitions
        self.network.topology.children.append(
            part := Rhomboid(
                name="dud_layer2", origin=[0, 0, 120], dimensions=[200, 200, 80]
            )
        )
        self.network.resize()
        placement = PlacementDud(
            name="dud",
            strategy="PlacementDud",
            partitions=[part],
            cell_types=[self.network.cell_types["cell_rel_count"]],
        )
        self.network.placement["dud3"] = placement
        indic = placement.get_indicators()["cell_rel_count"]
        # the target is 40 * 0.5 distributed in 4 chunks
        for chunk in part.to_chunks(np.array([100, 100, 100])):
            self.assertEqual(
                40 * 0.5 / 4, indic.guess(_chunk(chunk[0], chunk[1], chunk[2]))
            )

    def test_local_count_ratio(self):
        # Test placement with local count ratio in overlapping partitions
        # dud_layer2 overlaps with dud_layer
        self.network.topology.children[0].thickness = 200.0
        self.network.topology.children.append(
            part := Rhomboid(
                name="dud_layer2", origin=[100, 0, 0], dimensions=[100, 200, 100]
            )
        )
        self.network.resize()
        self.network.cell_types["cell_rel_count"].spatial.count_ratio = None
        self.network.cell_types["cell_rel_count"].spatial.local_count_ratio = 2.0
        placement = PlacementDud(
            name="dud",
            strategy="PlacementDud",
            partitions=[part],
            cell_types=[self.network.cell_types["cell_rel_count"]],
        )
        self.network.placement["dud3"] = placement
        indic = placement.get_indicators()["cell_rel_count"]
        # 2 chunks fully overlapping with the 8 from the target
        for chunk in part.to_chunks(np.array([100, 100, 100])):
            self.assertEqual(
                40 / 8 * 2.0, indic.guess(_chunk(chunk[0], chunk[1], chunk[2]))
            )

    def test_negative_guess_count(self):
        self.placement = single_layer_placement(
            self.network, offset=np.array([-300.0, -300.0, -300.0])
        )
        indicators = self.placement.get_indicators()
        dud_ind = indicators["cell_w_count"]
        bottom_ratio = 1 / 1.2
        bottom = 40 * bottom_ratio / 4
        top_ratio = 0.2 / 1.2
        top = 40 * top_ratio / 4
        for x, y, z in ((-3, -3, -3), (-3, -2, -3), (-2, -3, -3), (-2, -2, -3)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(bottom) <= guess <= np.ceil(bottom))
        for x, y, z in ((-3, -3, -2), (-3, -2, -2), (-2, -3, -2), (-2, -2, -2)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertTrue(np.floor(top) <= guess <= np.ceil(top))
        for x, y, z in ((0, 0, -1), (0, 0, 0), (2, 0, 0), (1, -3, 1)):
            with self.subTest(x=x, y=y, z=z):
                guess = dud_ind.guess(_chunk(x, y, z))
                self.assertEqual(0, guess)


@unittest.skipIf(MPI.get_size() > 1, "Skipped during parallel testing.")
class TestPlacementStrategies(
    RandomStorageFixture, NumpyTestCase, unittest.TestCase, engine_name="hdf5"
):
    def test_random_placement(self):
        cfg = get_test_config("single")
        network = Scaffold(cfg, self.storage)
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertEqual(40, len(ps), "fixed count random placement broken")

    def test_regression_issue_879(self):
        """
        If different partitions share chunks, these chunks should be dealt
        only once by the placement strategy.
        """
        cfg = get_test_config("single")
        cfg.partitions.add("test_layer2", {"thickness": 50.0})
        cfg.placement["test_placement"] = dict(
            strategy="bsb.placement.RandomPlacement",
            cell_types=["test_cell"],
            partitions=["test_layer", "test_layer2"],
        )
        network = Scaffold(cfg, self.storage)
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertEqual(
            40, len(ps), "multi partitions fixed count random placement broken"
        )

    def test_fixed_pos(self):
        cfg = Configuration.default(
            cell_types=dict(test_cell=dict(spatial=dict(radius=2, count=100))),
            placement=dict(
                ch4_c25=dict(
                    strategy="bsb.placement.strategy.FixedPositions",
                    partitions=[],
                    cell_types=["test_cell"],
                )
            ),
        )
        cs = cfg.network.chunk_size
        c4 = [
            Chunk((0, 0, 0), cs),
            Chunk((0, 0, 1), cs),
            Chunk((1, 0, 0), cs),
            Chunk((1, 0, 1), cs),
        ]
        cfg.placement.ch4_c25.positions = pos = MPI.bcast(
            np.vstack(tuple(c * cs + np.random.random((25, 3)) * cs for c in c4))
        )
        network = Scaffold(cfg, self.storage)
        network.compile()
        ps = network.get_placement_set("test_cell")
        pos_sort = pos[np.argsort(pos[:, 0])]
        pspos = ps.load_positions()
        pspos_sort = pspos[np.argsort(pspos[:, 0])]
        self.assertClose(pos_sort, pspos_sort, "expected fixed positions")

    def test_parallel_arrays(self):
        cfg = get_test_config("single")
        network = Scaffold(cfg, self.storage)
        cfg.placement["test_placement"] = dict(
            strategy="bsb.placement.ParallelArrayPlacement",
            cell_types=["test_cell"],
            partitions=["test_layer"],
            spacing_x=50,
            angle=0,
        )
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertEqual(39, len(ps), "fixed count parallel array placement broken")
        pos = ps.load_positions()
        self.assertAll(pos[:, 1] <= cfg.partitions.test_layer.data.mdc[1], "not in layer")
        self.assertAll(pos[:, 1] >= cfg.partitions.test_layer.data.ldc[1], "not in layer")

    def test_packed_arrays(self):
        cfg = get_test_config("single")
        network = Scaffold(cfg, self.storage)
        cfg.placement["test_placement"] = dict(
            strategy="bsb.placement.ParallelArrayPlacement",
            cell_types=["test_cell"],
            partitions=["test_layer"],
            spacing_x=150,
            angle=0,
        )
        with self.assertRaises(WorkflowError):
            network.compile(clear=True)

    def test_wrong_angles(self):
        cfg = get_test_config("single")
        network = Scaffold(cfg, self.storage)
        with self.assertRaises(BootError):
            cfg.placement["test_placement"] = dict(
                strategy="bsb.placement.ParallelArrayPlacement",
                cell_types=["test_cell"],
                partitions=["test_layer"],
                spacing_x=50,
                angle=90,
            )
        with self.assertRaises(BootError):
            cfg.placement["test_placement"] = dict(
                strategy="bsb.placement.ParallelArrayPlacement",
                cell_types=["test_cell"],
                partitions=["test_layer"],
                spacing_x=50,
                angle=-450,
            )

    def test_regression_issue_889(self):
        cfg = Configuration.default(
            regions={
                "cerebellar_cortex": {"type": "group", "children": ["purkinje_layer"]}
            },
            partitions={
                "purkinje_layer": {
                    "type": "rhomboid",
                    "origin": [100, 100, 0],
                    "dimensions": [100, 100, 15],
                }
            },
            cell_types={
                "purkinje_cell": {
                    "spatial": {"planar_density": 0.00045, "radius": 7.5},
                }
            },
            placement={
                "purkinje_layer_placement": {
                    "strategy": "bsb.placement.ParallelArrayPlacement",
                    "partitions": ["purkinje_layer"],
                    "cell_types": ["purkinje_cell"],
                    "spacing_x": 50,
                    "angle": 0,
                }
            },
        )
        network = Scaffold(cfg, self.storage)
        network.compile(clear=True)
        ps = network.get_placement_set("purkinje_cell")
        self.assertEqual(4, len(ps), "parallel array placement with offset is broken")


class TestVoxelDensities(RandomStorageFixture, unittest.TestCase, engine_name="hdf5"):
    def test_particle_vd(self):
        cfg = Configuration.default(
            cell_types=dict(
                test_cell=CellType(spatial=dict(radius=2, density=2, density_key="inhib"))
            ),
            regions=dict(test_region=dict(children=["test_part"])),
            partitions=dict(test_part=dict(type="test")),
            placement=dict(
                voxel_density=dict(
                    strategy="bsb.placement.RandomPlacement",
                    partitions=["test_part"],
                    cell_types=["test_cell"],
                )
            ),
        )
        network = Scaffold(cfg, self.storage)
        counts = network.placement.voxel_density.get_indicators()["test_cell"].guess(
            chunk=Chunk([0, 0, 0], [100, 100, 100]),
            voxels=network.partitions.test_part.vs,
        )
        self.assertEqual(4, len(counts), "should have vector of counts per voxel")
        # test rounded down values
        self.assertTrue(np.allclose([78, 15, 7, 26], counts, atol=1), "densities incorr")
        network.compile(clear=True)
        ps = network.get_placement_set("test_cell")
        self.assertGreater(len(ps), 125)  # rounded down values -1
        self.assertLess(len(ps), 132)  # rounded up values + 1

    def _config_packing_fact(self):
        return Configuration.default(
            network={
                "x": 20.0,
                "y": 20.0,
                "z": 5.0,
                "chunk_size": [20, 10, 20],  # at least two chunks
            },
            partitions={
                "first_layer": {"thickness": 5.0},
            },
            cell_types=dict(
                test_cell=dict(spatial=dict(radius=1.5, count=100)),
                test_cell2=dict(
                    spatial=dict(radius=2, relative_to="test_cell", count_ratio=0.05)
                ),
            ),
            placement=dict(
                test_place2=dict(
                    strategy="bsb.placement.RandomPlacement",
                    partitions=["first_layer"],
                    cell_types=["test_cell2"],
                ),
                ch4_c25=dict(
                    strategy="bsb.placement.RandomPlacement",
                    partitions=["first_layer"],
                    cell_types=["test_cell"],
                ),
            ),
        )

    def test_packing_factor_error1(self):
        cfg = self._config_packing_fact()
        network = Scaffold(cfg, self.storage)
        with self.assertRaises(WorkflowError):
            network.compile(clear=True)

    def test_packing_factor_error2(self):
        cfg = self._config_packing_fact()
        cfg.cell_types["test_cell"] = dict(spatial=dict(radius=1.3, count=100))
        network = Scaffold(cfg, self.storage)
        with self.assertRaises(WorkflowError):
            network.compile(clear=True)

    @skip_parallel
    def test_packing_factor_warning(self):
        """
        Test that particle placement warns for high density packing. Skipped during parallel because the warning
        is raised on a worker and can't be asserted on all nodes.
        """
        cfg = self._config_packing_fact()
        cfg.cell_types["test_cell"] = dict(spatial=dict(radius=1, count=100))
        network = Scaffold(cfg, self.storage)
        with self.assertWarns(PackingWarning):
            network.compile(clear=True)


class VoxelParticleTest(Partition, classmap_entry="test"):
    vs = VoxelSet(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ],
        25,
        data=VoxelData(
            np.array(
                [
                    [0.005, 0.003],
                    [0.001, 0.004],
                    [0.0005, 0.0055],
                    [0.0017, 0.0033],
                ]
            ),
            keys=["inhib", "excit"],
        ),
    )

    def to_chunks(self, chunk_size):
        return [Chunk([0, 0, 0], chunk_size)]

    def chunk_to_voxels(self, chunk):
        return self.vs

    def get_layout(self, hint):
        return hint.copy()

    # Noop city bish, noop noop city bish
    surface = volume = scale = translate = rotate = lambda self, smth: 5
