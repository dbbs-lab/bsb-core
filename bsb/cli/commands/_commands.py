"""
Contains builtin commands.
"""

from . import BaseCommand
from ...option import BsbOption
from ...exceptions import *
from . import _projects


class XScale(BsbOption, name="x", cli=("x",), env=("BSB_CONFIG_NETWORK_X",)):
    pass


class YScale(BsbOption, name="y", cli=("y",), env=("BSB_CONFIG_NETWORK_Y",)):
    pass


class ZScale(BsbOption, name="z", cli=("z",), env=("BSB_CONFIG_NETWORK_Z",)):
    pass


class Skip(BsbOption, name="skip", cli=("skip",), env=("BSB_SELECTION_SKIP",), list=True):
    pass


class Only(BsbOption, name="only", cli=("only",), env=("BSB_SELECTION_ONLY",), list=True):
    pass


class Append(
    BsbOption, name="append", cli=("append", "a"), env=("BSB_APPEND_MODE",), flag=True
):
    pass


class Redo(BsbOption, name="redo", cli=("redo", "r"), env=("BSB_REDO_MODE",), flag=True):
    pass


class Clear(
    BsbOption, name="clear", cli=("clear", "w"), env=("BSB_CLEAR_MODE",), flag=True
):
    pass


class Output(BsbOption, name="output", cli=("output", "o"), env=("BSB_OUTPUT_FILE",)):
    pass


class Plot(
    BsbOption, name="plot", cli=("plot", "p"), env=("BSB_PLOT_NETWORK",), flag=True
):
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


class ConfigOption(
    BsbOption,
    name="config",
    cli=("c", "config"),
    env=("BSB_CONFIG_FILE",),  # TODO: project=("config_file")
):
    """
    Specify the config file to use when creating new networks through the CLI.
    """

    def get_default(self):
        return "network_configuration.json"


class MakeConfigCommand(BaseCommand, name="make-config"):
    def handler(self, context):
        from ...config import copy_template

        args = context.arguments
        copy_template(args.template, args.output, path=args.path or ())

    def get_options(self):
        return {}

    def add_parser_arguments(self, parser):
        parser.add_argument("template", nargs="?", default="template.json")
        parser.add_argument("output", nargs="?", default="network_configuration.json")
        parser.add_argument(
            "--path",
            help="Additional paths to search for config templates",
            action="extend",
            nargs="+",
            default=False,
        )


class BsbCompile(BaseCommand, name="compile"):
    def handler(self, context):
        from ...config import from_json
        from ...core import Scaffold

        cfg = from_json(context.config)
        # Bootstrap the scaffold and clear the storage if not in append mode
        network = Scaffold(cfg)
        network.resize(context.x, context.y, context.z)
        network.compile(
            skip_placement=context.skip_placement,
            skip_after_placement=context.skip_after_placement,
            skip_connectivity=context.skip_connectivity,
            skip_after_connectivity=context.skip_after_connectivity,
            only=context.only,
            skip=context.skip,
            clear=context.clear,
            force=context.force,
            append=context.append,
            redo=context.redo,
        )

        if context.plot:
            from bsb.plotting import plot_network

            plot_network(network)

    def get_options(self):
        return {
            "x": XScale(),
            "y": YScale(),
            "z": ZScale(),
            "skip": Skip(),
            "only": Only(),
            "config": ConfigOption(positional=True),
            "no_placement": SkipPlacement(),
            "no_after_placement": SkipAfterPlacement(),
            "no_connectivity": SkipConnectivity(),
            "no_after_connectivity": SkipAfterConnectivity(),
            "append": Append(),
            "redo": Redo(),
            "clear": Clear(),
            "plot": Plot(),
            "output": Output(),
        }

    def add_parser_arguments(self, parser):
        pass


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
