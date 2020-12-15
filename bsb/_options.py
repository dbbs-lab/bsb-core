from .option import BsbOption
from .reporting import report


class VerbosityOption(
    BsbOption,
    name="verbosity",
    cli=("v", "verbosity"),
    env=("BSB_VERBOSITY",),
    script=("verbosity",),
):
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
    def get_default(self):
        return 1

    def action(self):
        from . import __version__

        report(__version__, level=1)


class ConfigOption(
    BsbOption, name="config", cli=("c", "config"), env=("BSB_CONFIG_FILE",)
):
    def get_default(self):
        return 1


def verbosity():
    return VerbosityOption


def config():
    return ConfigOption


def version():
    return VersionFlag
