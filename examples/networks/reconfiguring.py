# Example that shows how to overwrite the configuration inside of an existing network
from bsb import Configuration, from_storage

network = from_storage("network.hdf5")
new_config = Configuration.default()
network.storage.store_active_config(new_config)
