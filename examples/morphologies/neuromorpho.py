from bsb.core import Scaffold
from bsb.config import from_json

cfg = from_json("network_configuration.json")
network = Scaffold(cfg)
# The selectors tell us which morphologies are configured
selectors = cfg.cell_types.top_type.spatial.morphologies
# And we pass them into the `select` function to load their info
info = network.morphologies.select(*selectors)
print("Fetched info of:", [loader.name for loader in info])
# Print out the NeuroMorpho metadata of the first morphology
print(info[0].get_meta())
# Load the first morphology
morpho0 = info[0].load()
print(morpho0)
