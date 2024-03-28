import abc


class ConfigurationParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, content, path=None):
        pass

    @abc.abstractmethod
    def generate(self, tree, pretty=False):
        pass


def get_parser_classes():
    from ..plugins import discover

    return discover("config.parsers")


def get_parser(parser):
    return get_parser_classes()[parser]()


__all__ = ["ConfigurationParser", "get_parser", "get_parser_classes"]
