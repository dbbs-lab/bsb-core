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
    project=("verbosity",),
    env=("BSB_VERBOSITY",),
    script=("verbosity",),
):
    """
    Set the verbosity of the package. Verbosity 0 is completely silent, 1 is default,
    2 is verbose, 3 is progress and 4 is debug.
    """

    def setter(self, value):
        return int(value)

    def getter(self, value):
        return int(value)

    def get_default(self):
        return 1


class ForceFlag(
    BsbOption,
    name="force",
    cli=("f", "force"),
    env=("BSB_FOOTGUN_MODE",),
    script=("sudo",),
    flag=True,
):
    """
    Enable sudo mode. Will execute destructive actions without confirmation, error or user
    interaction.
    """

    def setter(self, value):
        return bool(value)

    def get_default(self):
        return False


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
        report("bsb " + str(self.get()), level=1)


class ConfigOption(
    BsbOption,
    name="config",
    cli=("c", "config"),
    script=("config",),
    project=("config",),
    env=("BSB_CONFIG_FILE",),
):
    """
    Specify the config file to use when creating new networks through the CLI.
    """

    def get_default(self):
        return "network_configuration.json"


class ProfilingOption(
    BsbOption,
    name="profiling",
    cli=("pr", "profiling"),
    project=("profiling",),
    script=("profiling",),
    env=("BSB_PROFILING",),
    flag=True,
):
    """
    Enables profiling.
    """

    def setter(self, value):
        from .profiling import activate_session, get_active_session

        if value:
            activate_session()
        else:
            get_active_session().stop()
        return bool(value)

    def getter(self, value):
        return bool(value)

    def get_default(self):
        return False


class DebugPoolFlag(
    BsbOption,
    name="debug_pool",
    cli=("dp", "debug_pool"),
    project=("debug_pool",),
    env=("BSB_DEBUG_POOL",),
    script=("debug_pool",),
    flag=True,
):
    """
    Debug job pools
    """

    def setter(self, value):
        return bool(value)

    def get_default(self):
        return False


def verbosity():
    return VerbosityOption


def version():
    return VersionFlag


def sudo():
    return ForceFlag


def config():
    return ConfigOption


def profiling():
    return ProfilingOption


def debug_pool():
    return DebugPoolFlag
