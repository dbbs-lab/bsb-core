import abc
import functools

from ..exceptions import PluginError


class ConfigurationParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, content, path=None):
        pass

    @abc.abstractmethod
    def generate(self, tree, pretty=False):
        pass


@functools.cache
def get_configuration_parser_classes():
    from ..plugins import discover

    return discover("config.parsers")


def get_configuration_parser(parser, **kwargs):
    """
    Create an instance of a configuration parser that can parse configuration
    strings into configuration trees, or serialize trees into strings.

    Configuration trees can be cast into Configuration objects.
    """
    parsers = get_configuration_parser_classes()
    if parser not in parsers:
        raise PluginError(f"Configuration parser '{parser}' not found")
    return parsers[parser](**kwargs)


__all__ = [
    "ConfigurationParser",
    "get_configuration_parser",
    "get_configuration_parser_classes",
]
