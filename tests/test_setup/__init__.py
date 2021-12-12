import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig

scaffold_lookup = {}
mr_path = os.path.join(os.path.dirname(__file__), "..", "morphologies.h5")
mr_top_path = os.path.join(os.path.dirname(__file__), "..", "..", "morphologies.h5")
mr_rot_path = os.path.join(os.path.dirname(__file__), "..", "morpho_rotated.h5")
rotations_step = [30, 60]


def get_test_network(x=None, z=None):
    t = tuple([x, z])
    if not t in scaffold_lookup:
        _create_test_network(x, z)

    scaffold = from_hdf5(scaffold_lookup[t])
    return scaffold


def get_tiny_network():
    scaffold_filename = "_test_tiny_network.hdf5"
    scaffold_path = os.path.join(os.path.dirname(__file__), "..", scaffold_filename)
    if not os.path.exists(scaffold_path):
        config = JSONConfig(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "configs",
                "3_9_mouse.json",
            )
        )
        config.output_formatter.file = scaffold_path
        scaffold = Scaffold(config)
        mf_ids = scaffold.create_entities(
            scaffold.configuration.cell_types["mossy_fibers"], 3
        )
        glom_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["glomerulus"],
            None,
            np.array([[0, 0, 0]] * 12),
        )
        grc_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["granule_cell"],
            None,
            np.array([[0, 0, 0]] * 12),
        )
        goc_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["golgi_cell"],
            None,
            np.array([[0, 0, 0]] * 2),
        )
        pc_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["purkinje_cell"],
            None,
            np.array([[0, 0, 0]]),
        )
        sc_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["stellate_cell"],
            None,
            np.array([[0, 0, 0]] * 3),
        )
        bc_ids = scaffold.place_cells(
            scaffold.configuration.cell_types["basket_cell"],
            None,
            np.array([[0, 0, 0]] * 3),
        )

        scaffold.connect_cells(
            scaffold.configuration.connection_types["mossy_to_glomerulus"],
            np.array(
                [
                    [mf_ids[0], glom_ids[0 * 4]],
                    [mf_ids[0], glom_ids[0 * 4 + 1]],
                    [mf_ids[0], glom_ids[0 * 4 + 2]],
                    [mf_ids[0], glom_ids[0 * 4 + 3]],
                    [mf_ids[1], glom_ids[1 * 4]],
                    [mf_ids[1], glom_ids[1 * 4 + 1]],
                    [mf_ids[1], glom_ids[1 * 4 + 2]],
                    [mf_ids[1], glom_ids[1 * 4 + 3]],
                    [mf_ids[2], glom_ids[2 * 4]],
                    [mf_ids[2], glom_ids[2 * 4 + 1]],
                    [mf_ids[2], glom_ids[2 * 4 + 2]],
                    [mf_ids[2], glom_ids[2 * 4 + 3]],
                ]
            ),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["glomerulus_to_granule"],
            np.array(
                [
                    [glom_ids[0], grc_ids[0]],
                    [glom_ids[0], grc_ids[1]],
                    [glom_ids[0], grc_ids[2]],
                    [glom_ids[0], grc_ids[3]],
                    [glom_ids[0], grc_ids[4]],
                    [glom_ids[1], grc_ids[0]],
                    [glom_ids[1], grc_ids[1]],
                    [glom_ids[1], grc_ids[2]],
                    [glom_ids[1], grc_ids[3]],
                    [glom_ids[1], grc_ids[4]],
                    [glom_ids[2], grc_ids[0]],
                    [glom_ids[2], grc_ids[1]],
                    [glom_ids[2], grc_ids[2]],
                    [glom_ids[2], grc_ids[3]],
                    [glom_ids[2], grc_ids[4]],
                    [glom_ids[3], grc_ids[5]],
                    [glom_ids[3], grc_ids[6]],
                    [glom_ids[3], grc_ids[7]],
                    [glom_ids[3], grc_ids[8]],
                    [glom_ids[3], grc_ids[9]],
                    [glom_ids[4], grc_ids[5]],
                    [glom_ids[4], grc_ids[6]],
                    [glom_ids[4], grc_ids[7]],
                    [glom_ids[4], grc_ids[8]],
                    [glom_ids[4], grc_ids[9]],
                    [glom_ids[5], grc_ids[5]],
                    [glom_ids[5], grc_ids[6]],
                    [glom_ids[5], grc_ids[7]],
                    [glom_ids[5], grc_ids[8]],
                    [glom_ids[5], grc_ids[9]],
                    [glom_ids[6], grc_ids[5]],
                    [glom_ids[6], grc_ids[6]],
                    [glom_ids[6], grc_ids[7]],
                    [glom_ids[6], grc_ids[8]],
                    [glom_ids[6], grc_ids[9]],
                    [glom_ids[7], grc_ids[10]],
                    [glom_ids[8], grc_ids[11]],
                    [glom_ids[9], grc_ids[10]],
                    [glom_ids[10], grc_ids[0]],
                    [glom_ids[11], grc_ids[1]],
                ]
            ),
            compartments=(
                c := np.array(
                    [
                        *([[-1, 113]] * 5),
                        *([[-1, 122]] * 5),
                        *([[-1, 131]] * 5),
                        *([[-1, 113]] * 5),
                        *([[-1, 122]] * 5),
                        *([[-1, 131]] * 5),
                        *([[-1, 140]] * 5),
                        *([[-1, 113]] * 2),
                        [-1, 122],
                        [-1, 140],
                        [-1, 140],
                    ]
                )
            ),
            morphologies=np.array([["GranuleCell"] * 2] * c.shape[0]),
        )

        scaffold.connect_cells(
            scaffold.configuration.connection_types["parallel_fiber_to_golgi"],
            np.array(
                [
                    [grc_ids[0], goc_ids[0]],
                    [grc_ids[1], goc_ids[0]],
                    [grc_ids[1], goc_ids[0]],
                    [grc_ids[3], goc_ids[0]],
                    [grc_ids[4], goc_ids[0]],
                    [grc_ids[5], goc_ids[0]],
                    [grc_ids[6], goc_ids[0]],
                    [grc_ids[7], goc_ids[0]],
                    [grc_ids[8], goc_ids[0]],
                    [grc_ids[9], goc_ids[0]],
                    [grc_ids[10], goc_ids[0]],
                    [grc_ids[11], goc_ids[0]],
                    [grc_ids[0], goc_ids[1]],
                    [grc_ids[1], goc_ids[1]],
                    [grc_ids[1], goc_ids[1]],
                    [grc_ids[3], goc_ids[1]],
                    [grc_ids[4], goc_ids[1]],
                    [grc_ids[5], goc_ids[1]],
                    [grc_ids[6], goc_ids[1]],
                    [grc_ids[7], goc_ids[1]],
                    [grc_ids[8], goc_ids[1]],
                    [grc_ids[9], goc_ids[1]],
                    [grc_ids[10], goc_ids[1]],
                    [grc_ids[11], goc_ids[1]],
                ]
            ),
            compartments=np.array(
                [
                    *([[58, 2930]] * 6),
                    *([[59, 2929]] * 6),
                    *([[60, 2928]] * 6),
                    *([[61, 2927]] * 6),
                ]
            ),
            morphologies=np.array([["GranuleCell", "GolgiCell"]] * 24),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["ascending_axon_to_golgi"],
            np.array(
                [
                    [grc_ids[0], goc_ids[1]],
                    [grc_ids[1], goc_ids[1]],
                    [grc_ids[1], goc_ids[1]],
                    [grc_ids[3], goc_ids[1]],
                    [grc_ids[4], goc_ids[1]],
                    [grc_ids[5], goc_ids[1]],
                    [grc_ids[6], goc_ids[1]],
                    [grc_ids[7], goc_ids[1]],
                    [grc_ids[8], goc_ids[1]],
                    [grc_ids[9], goc_ids[1]],
                    [grc_ids[10], goc_ids[1]],
                    [grc_ids[11], goc_ids[1]],
                    [grc_ids[0], goc_ids[0]],
                    [grc_ids[1], goc_ids[0]],
                    [grc_ids[1], goc_ids[0]],
                    [grc_ids[3], goc_ids[0]],
                    [grc_ids[4], goc_ids[0]],
                    [grc_ids[5], goc_ids[0]],
                    [grc_ids[6], goc_ids[0]],
                    [grc_ids[7], goc_ids[0]],
                    [grc_ids[8], goc_ids[0]],
                    [grc_ids[9], goc_ids[0]],
                    [grc_ids[10], goc_ids[0]],
                    [grc_ids[11], goc_ids[0]],
                ]
            ),
            compartments=np.array(
                [
                    *([[58, 2931]] * 6),
                    *([[59, 2932]] * 6),
                    *([[60, 2933]] * 6),
                    *([[61, 2802]] * 6),
                ]
            ),
            morphologies=np.array([["GranuleCell", "GolgiCell"]] * 24),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["golgi_to_golgi"],
            np.array(
                [*([[goc_ids[0], goc_ids[1]]] * 100), *([[goc_ids[1], goc_ids[0]]] * 100)]
            ),
            compartments=np.tile(
                np.column_stack((np.arange(396, 496), np.arange(495, 395, -1))), (2, 1)
            ),
            morphologies=np.tile(np.array(["GolgiCell"]), (200, 2)),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["gap_goc"],
            np.array(
                [*([[goc_ids[0], goc_ids[1]]] * 5), *([[goc_ids[1], goc_ids[0]]] * 5)]
            ),
            compartments=np.ones((10, 2)) * 395,
            morphologies=np.tile(["GolgiCell"], (10, 2)),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["parallel_fiber_to_stellate"],
            np.column_stack(
                (
                    np.tile(grc_ids, len(sc_ids)),
                    np.tile(sc_ids, (len(grc_ids), 1)).T.flatten(),
                )
            ),
            compartments=np.tile([80, 2580], (len(grc_ids) * len(sc_ids), 1)),
            morphologies=np.tile(
                ["GranuleCell", "StellateCell"], (len(grc_ids) * len(sc_ids), 1)
            ),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["parallel_fiber_to_basket"],
            np.column_stack(
                (
                    np.tile(grc_ids, len(bc_ids)),
                    np.tile(bc_ids, (len(grc_ids), 1)).T.flatten(),
                )
            ),
            compartments=np.tile([79, 2740], (len(grc_ids) * len(bc_ids), 1)),
            morphologies=np.tile(
                ["GranuleCell", "BasketCell"], (len(grc_ids) * len(bc_ids), 1)
            ),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["parallel_fiber_to_purkinje"],
            np.column_stack(
                (
                    np.tile(grc_ids, len(pc_ids)),
                    np.tile(pc_ids, (len(grc_ids), 1)).T.flatten(),
                )
            ),
            compartments=np.tile([81, 2371], (len(grc_ids) * len(pc_ids), 1)),
            morphologies=np.tile(
                ["GranuleCell", "PurkinjeCell"], (len(grc_ids) * len(pc_ids), 1)
            ),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["stellate_to_purkinje"],
            np.column_stack(
                (
                    np.tile(sc_ids, len(pc_ids)),
                    np.tile(pc_ids, (len(sc_ids), 1)).T.flatten(),
                )
            ),
            compartments=np.tile([2870, 2875], (len(sc_ids) * len(pc_ids), 1)),
            morphologies=np.tile(
                ["StellateCell", "PurkinjeCell"], (len(sc_ids) * len(pc_ids), 1)
            ),
        )
        scaffold.connect_cells(
            scaffold.configuration.connection_types["basket_to_purkinje"],
            np.column_stack(
                (
                    np.tile(bc_ids, len(pc_ids)),
                    np.tile(pc_ids, (len(bc_ids), 1)).T.flatten(),
                )
            ),
            compartments=np.tile([6129, 10], (len(bc_ids) * len(pc_ids), 1)),
            morphologies=np.tile(
                ["BasketCell", "PurkinjeCell"], (len(bc_ids) * len(pc_ids), 1)
            ),
        )
        scaffold.compile_output()
    else:
        scaffold = from_hdf5(scaffold_path)


def _create_test_network(*dimensions):
    scaffold_filename = f"_test_network_{dimensions[0]}_{dimensions[1]}.hdf5"
    scaffold_lookup[tuple(dimensions)] = scaffold_filename
    scaffold_path = os.path.join(os.path.dirname(__file__), "..", scaffold_filename)
    if not os.path.exists(scaffold_path):
        # Fetch the default configuration file.
        config = JSONConfig(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "configs",
                "legacy_mouse_cerebellum.json",
            )
        )
        config.resize(*dimensions)
        print("Initializing {} by {} network for testing.".format(config.X, config.Z))
        # Resize the simulation volume to the specified dimensions
        # Set output filename
        config.output_formatter.file = scaffold_filename
        # Bootstrap the scaffold and create the network.
        scaffold = Scaffold(config)
        scaffold.compile_network()


def prep_morphologies():
    if not os.path.exists(mr_path):
        from bsb.output import MorphologyRepository as MR
        import dbbs_models

        mr = MR(mr_path)
        mr.import_arbz_module(dbbs_models)
        import shutil

        shutil.copyfile(mr_path, mr_top_path)


def prep_rotations():
    if not os.path.exists(mr_rot_path):
        from bsb.output import MorphologyRepository, MorphologyCache
        import dbbs_models

        mr = MorphologyRepository(mr_rot_path)
        mr.get_handle("w")
        mr.import_arbz("GranuleCell", dbbs_models.GranuleCell)
        mr.import_arbz("GolgiCell", dbbs_models.GolgiCell)
        mr.import_arbz("GolgiCell_A", dbbs_models.GolgiCell)
        mc = MorphologyCache(mr)
        mc.rotate_all_morphologies(rotations_step[0], rotations_step[1])
