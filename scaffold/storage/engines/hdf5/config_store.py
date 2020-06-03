from ...interfaces import ConfigStore as IConfigStore
from ....helpers import get_configurable_class, get_qualified_class_name
from .... import __version__
import numpy as np


class ConfigStore(IConfigStore):
    def __init__(self, handler):
        self._handler = handler

    def load(self):
        with self._handler.open("r") as f:
            resource = f()
            # Get the serialized configuration details
            config_class_name = resource.attrs["configuration_class"]
            config_string = resource.attrs["configuration_string"]
            return get_configurable_class(config_class_name)(stream=config_string)

    def store(self, config):
        print("Storing config")
        with self._handler.open("a") as f:
            f = f()
            f.attrs["version"] = __version__
            f.attrs["configuration_name"] = config._name
            f.attrs["configuration_type"] = config._type
            f.attrs["configuration_class"] = get_qualified_class_name(config)
            # REALLY BAD HACK: This is to cover up #222 in the test networks during unit testing.
            f.attrs["configuration_string"] = config._raw.replace(
                '"simulation_volume_x": 400.0', '"simulation_volume_x": ' + str(config.X)
            ).replace(
                '"simulation_volume_z": 400.0', '"simulation_volume_z": ' + str(config.Z)
            )
