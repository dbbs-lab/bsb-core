"""
This module registers all the options that the BSB provides out of the box. They are
registered using plugin registration. See `setup.py` for the setuptools metadata.
"""

from .option import BsbOption
from .reporting import report


class VerbosityOption(
    BsbOption,
    name="verbosity",
    cli=("v", "verbosity"),
    env=("BSB_VERBOSITY",),
    script=("verbosity",),
):
    """
    Set the verbosity of the package. Verbosity 0 is completely silent, 1 is default,
    2 is verbose, 3 is progress and 4 is debug.
    """

    def setter(self, value):
        return int(value)

    def get_default(self):
        return 1


class VersionFlag(
    BsbOption,
    name="version",
    script=("version",),
    cli=("version",),
    readonly=True,
    action=True,
):
    """
    Return the version of the package.
    """

    def get_default(self):
        from . import __version__

        return __version__

    def action(self, namespace):
        report(self.get(), level=1)


def verbosity():
    return VerbosityOption


def version():
    return VersionFlag
