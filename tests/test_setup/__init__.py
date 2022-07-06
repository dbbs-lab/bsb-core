import os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from bsb.core import Scaffold, from_hdf5

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


def get_config(file):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "configs",
            file + (".json" if not file.endswith(".json") else ""),
        )
    )


def get_morphology(file):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "morphologies",
            file,
        )
    )
