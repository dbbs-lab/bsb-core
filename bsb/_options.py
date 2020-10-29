from .option import BsbOption


class VerbosityOption(
    BsbOption,
    name="verbosity",
    cli=("v", "verbosity"),
    env=("BSB_VERBOSITY",),
    script=("verbosity",),
):
    def get_default(self):
        return 1


class ConfigOption(
    BsbOption, name="config", cli=("c", "config"), env=("BSB_CONFIG_FILE",)
):
    def get_default(self):
        return 1


def verbosity():
    return VerbosityOption


def config():
    return ConfigOption
