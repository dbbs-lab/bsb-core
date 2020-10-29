from . import BaseCommand
from ...option import BsbOption


class XScale(BsbOption, name="x", cli=("x",), env=("BSB_CONFIG_NETWORK_X",)):
    pass


class ZScale(BsbOption, name="z", cli=("z",), env=("BSB_CONFIG_NETWORK_Z",)):
    pass


class Skip(BsbOption, name="skip", cli=("skip",), env=("BSB_SELECTION_SKIP",), list=True):
    pass


class Only(BsbOption, name="only", cli=("only",), env=("BSB_SELECTION_ONLY",), list=True):
    pass


class SkipPlacement(
    BsbOption,
    name="skip_placement",
    cli=("np", "skip-placement"),
    env=("BSB_SKIP_PLACEMENT",),
    flag=True,
):
    pass


class SkipAfterPlacement(
    BsbOption,
    name="skip_after_placement",
    cli=("nap", "skip-after-placement"),
    env=("BSB_SKIP_AFTER_PLACEMENT",),
    flag=True,
):
    pass


class SkipConnectivity(
    BsbOption,
    name="skip_connectivity",
    cli=("nc", "skip-connectivity"),
    env=("BSB_SKIP_CONNECTIVITY",),
    flag=True,
):
    pass


class SkipAfterConnectivity(
    BsbOption,
    name="skip_after_connectivity",
    cli=("nac", "skip-after-connectivity"),
    env=("BSB_SKIP_AFTER_CONNECTIVITY",),
    flag=True,
):
    pass


class BsbCompile(BaseCommand, name="compile"):
    def handler(self, context):
        print(context.x)
        print(context.arguments)

    def get_options(self):
        return {
            "x": XScale(),
            "z": ZScale(),
            "skip": Skip(),
            "only": Only(),
            "no_placement": SkipPlacement(),
            "no_after_placement": SkipAfterPlacement(),
            "no_connectivity": SkipConnectivity(),
            "no_after_connectivity": SkipAfterConnectivity(),
        }

    def add_parser_arguments(self, parser):
        parser.add_argument("hello", help="positional")


class BsbSimulate(BaseCommand, name="simulate"):
    def handler(self, context):
        pass

    def get_options(self):
        return {"skip": Skip(), "only": Only()}

    def add_parser_arguments(self, parser):
        parser.add_argument("hello", help="positional")


def compile():
    return BsbCompile


def simulate():
    return BsbSimulate
