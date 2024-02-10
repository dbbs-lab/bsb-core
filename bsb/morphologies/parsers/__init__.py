import typing
from os import PathLike

from .parser import MorphologyParser


def parse_morphology_content(content: typing.Union[str, bytes], parser="bsb", **kwargs):
    return MorphologyParser(parser=parser, **kwargs).parse_content(content)


def parse_morphology_file(file: typing.Union[str, PathLike], parser="bsb", **kwargs):
    return MorphologyParser(parser=parser, **kwargs).parse(file)
