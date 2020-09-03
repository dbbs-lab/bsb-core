import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig

scaffold_lookup = {}


def get_test_network(x=None, z=None):
    t = tuple([x, z])
    if not t in scaffold_lookup:
        _create_test_network(x, z)

    scaffold = from_hdf5(scaffold_lookup[t])
    return scaffold


def _create_test_network(*dimensions):
    scaffold_filename = "_test_network_{}_{}.hdf5".format(dimensions[0], dimensions[1])
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
