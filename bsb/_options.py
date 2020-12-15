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

    def action(self):
        report(self.get_default(), level=1)


class ConfigOption(
    BsbOption, name="config", cli=("c", "config"), env=("BSB_CONFIG_FILE",)
):
    """
    Specify the config file to use when creating new networks through the CLI.
    """

    def get_default(self):
        return "template.json"


def verbosity():
    return VerbosityOption


def config():
    return ConfigOption


def version():
    return VersionFlag
