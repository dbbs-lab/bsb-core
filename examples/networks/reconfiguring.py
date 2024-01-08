# Example that shows how to overwrite the configuration inside of an existing network
from bsb.config import JSONConfig
from bsb.output import HDF5Formatter

config = JSONConfig("new_config.json")
HDF5Formatter.reconfigure("my_network.hdf5", config)
