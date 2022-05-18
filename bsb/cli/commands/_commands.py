"""
Contains builtin commands.
"""

from . import BaseCommand
from ...option import BsbOption
from ...exceptions import *
from ..._options import ConfigOption
from . import _projects
import itertools


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


def _flatten_arr_args(arr):
    if arr is None:
        return arr
    else:
        return list(itertools.chain.from_iterable(a.split(",") for a in arr))


class MakeConfigCommand(BaseCommand, name="make-config"):
    def handler(self, context):
        from ...config import copy_template

        args = context.arguments
        copy_template(args.template, args.output, path=args.path or ())

    def get_options(self):
        return {}

    def add_parser_arguments(self, parser):
        parser.add_argument("template", nargs="?", default="skeleton.json")
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
            only=_flatten_arr_args(context.only),
            skip=_flatten_arr_args(context.skip),
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
        return {
            "skip": Skip(),
            "only": Only(),
        }

    def add_parser_arguments(self, parser):
        pass


class CacheCommand(BaseCommand, name="cache"):  # pragma: nocover
    def handler(self, context):
        import shutil
        from datetime import datetime
        from ...storage._util import _cache_path

        if context.clear:
            shutil.rmtree(_cache_path)
            _cache_path.mkdir(parents=True, exist_ok=True)
            print("Cache cleared")
        else:
            _cache_path.mkdir(parents=True, exist_ok=True)
            files = [*_cache_path.iterdir()]
            maxlen = 5
            try:
                maxlen = max(maxlen, max(len(l.name) for l in files))
            except ValueError:
                print("Cache is empty")
            else:
                print(f"{'Files'.ljust(maxlen, ' ')}    Cached at\t\t\t    Size")
                total_mb = 0
                for f in files:
                    name = f.name.ljust(maxlen, " ")
                    stat = f.stat()
                    stamp = datetime.fromtimestamp(stat.st_mtime)
                    total_mb += (mb := stat.st_size / 1e6)
                    line = f"{name}    {stamp}    {mb:.2f}MB"
                    print(line)
                print(f"Total: {total_mb:.2f}MB".rjust(len(line)))

    def get_options(self):
        return {
            "clear": Clear(),
        }

    def add_parser_arguments(self, parser):
        pass
