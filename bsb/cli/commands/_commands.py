"""
Contains builtin commands.
"""

import itertools
import os
import pathlib
from uuid import uuid4

import errr

from ..._options import ConfigOption
from ...config import parse_configuration_file
from ...core import Scaffold, from_storage
from ...exceptions import NodeNotFoundError
from ...option import BsbOption
from ...reporting import report
from ...storage import open_storage
from . import BaseCommand


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


class IgnoreErrors(
    BsbOption,
    name="ignore_errors",
    cli=("ignore", "ignore-errors"),
    env=("BSB_IGNORE_ERRORS",),
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
        from ...config import copy_configuration_template

        args = context.arguments
        copy_configuration_template(args.template, args.output, path=args.path or ())

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
        cfg = parse_configuration_file(context.config)
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
            fail_fast=not context.ignore_errors,
        )

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
            "output": Output(),
            "ignore_errors": IgnoreErrors(),
        }

    def add_parser_arguments(self, parser):
        pass


class BsbReconfigure(BaseCommand, name="reconfigure"):
    def handler(self, context):
        cfg = parse_configuration_file(context.config)
        # Bootstrap the scaffold and clear the storage if not in append mode
        storage = open_storage(context.arguments.network)
        storage.store_active_config(cfg)

    def get_options(self):
        return {
            "config": ConfigOption(positional=True),
        }

    def add_parser_arguments(self, parser):
        parser.add_argument("network")


class BsbSimulate(BaseCommand, name="simulate"):
    def handler(self, context):
        network = from_storage(context.arguments.network)
        config_option = context.options["config"]
        sim_name = context.arguments.simulation
        extra_simulations = {}
        if config_option.is_set("cli"):
            extra_simulations = parse_configuration_file(context.config).simulations
            for name, sim in extra_simulations.items():
                if name not in network.simulations and name == sim_name:
                    network.simulations[sim_name] = sim
        root = pathlib.Path(getattr(context.arguments, "output_folder", "./"))
        if not root.is_dir() or not os.access(root, os.W_OK):
            return report(
                f"Output provided '{root.absolute()}' is not an existing directory with write access.",
                level=0,
            )
        try:
            result = network.run_simulation(sim_name)
        except NodeNotFoundError as e:
            append = ", " if len(network.simulations) else ""
            append += ", ".join(f"'{name}'" for name in extra_simulations.keys())
            errr.wrap(type(e), e, append=append)
        else:
            result.write(root / f"{uuid4()}.nio", "ow")

    def get_options(self):
        return {
            "skip": Skip(),
            "only": Only(),
        }

    def add_parser_arguments(self, parser):
        parser.add_argument("network")
        parser.add_argument("simulation")
        parser.add_argument("-o", "--output-folder")


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
                maxlen = max(maxlen, max(len(file.name) for file in files))
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
