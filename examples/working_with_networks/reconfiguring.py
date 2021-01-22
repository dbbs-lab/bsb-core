# Example that shows how to overwrite the configuration inside of an existing network
from bsb.output import HDF5Formatter
from bsb.config import JSONConfig

config = JSONConfig("new_config.json")
HDF5Formatter.reconfigure("my_network.hdf5", config)
