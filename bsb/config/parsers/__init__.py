from ._parser import Parser
from .json import JsonParser


def get_parser_classes():
    from ...plugins import discover

    return discover("config.parsers")


def get_parser(parser):
    return get_parser_classes()[parser]()
