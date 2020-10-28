from .option import BsbOption


class VerbosityOption(
    BsbOption, cli=("v", "verbosity"), env=("BSB_VERBOSITY",), script=("verbosity",)
):
    def get_default(self):
        return 1


def verbosity():
    return VerbosityOption
