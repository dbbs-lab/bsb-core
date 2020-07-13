from ...interfaces import ConfigStore as IConfigStore
from .... import __version__
from ....exceptions import *


class ConfigStore(IConfigStore):
    def __init__(self, handler):
        self._handler = handler

    def load(self):
        raise NotImplementedError("Under construction")

        from ... import config

        # Load an instance of the configuration parser that serialized it.
        parser_name = self.get_parser_name()
        # Throws a PluginError if that parser is not found.
        try:
            parser = config.get_parser(parser_name)
        except PluginError:
            raise PluginError(
                "Cannot parse configuration: The parser '{}' is not installed on this system.".format(
                    parser_name
                )
            ) from None
        # Read the ConfigStoreObject (cso) and cast it into a Configuration object (co)
        with self._handler.open("r") as f:
            resource = f()
            cso, meta = parser.parse(resource["content"][()])
            co = config.Configuration.__cast__(cso, None)
            co._parser_meta = meta
            return co

    def store(self, config):
        with self._handler.open("a") as f:
            # f = f()
            # f.attrs["version"] = __version__
            # f.attrs["configuration_name"] = config._name
            # f.attrs["configuration_type"] = config._type
            # f.attrs["configuration_class"] = get_qualified_class_name(config)
            # # REALLY BAD HACK: This is to cover up #222 in the test networks during unit testing.
            # f.attrs["configuration_string"] = config._raw.replace(
            #     '"simulation_volume_x": 400.0', '"simulation_volume_x": ' + str(config.X)
            # ).replace(
            #     '"simulation_volume_z": 400.0', '"simulation_volume_z": ' + str(config.Z)
            # )
            raise NotImplementedError("Under construction :3")

    def get_parser_name(self):
        with self._handler.open("a") as f:
            f = f()
            if not "configuration" in f:
                raise ConfigStoreError("No configuration stored, cannot get parser name.")
            return f["configuration"].attrs["parser"]
