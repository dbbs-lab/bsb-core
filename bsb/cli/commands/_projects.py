import pathlib

import toml

from ... import config
from ...reporting import report
from . import BaseCommand
from ... import plugins

import rlcompleter
import readline
import logging
import os


class SimpleCompleter:

    def __init__(self, options):
        self.options = sorted(options)

    def complete(self, text, state):
        response = None
        if state == 0:
            # This is the first time for this text,
            # so build a match list.
            if text:
                self.matches = [
                    s
                    for s in self.options
                    if s and s.startswith(text)
                ]
                logging.debug('%s matches: %s',
                              repr(text), self.matches)
            else:
                self.matches = self.options[:]
                logging.debug('(empty input) matches: %s',
                              self.matches)

        # Return the state'th item from the match list,
        # if we have that many.
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        logging.debug('complete(%s, %s) => %s',
                      repr(text), state, repr(response))
        return response

class ProjectNewCommand(BaseCommand, name="new"):
    def get_options(self):
        return {}

    def add_parser_arguments(self, parser):
        parser.add_argument("project_name", nargs="?", help="Project name", default="")
        parser.add_argument(
            "path", nargs="?", default=".", help="Location of the project"
        )
        parser.add_argument(
            "--quickstart", action="store_true", help="Start an example project"
        )
        parser.add_argument(
            "--json", action="store_true", help="Use JSON as configuration language"
        )
        parser.add_argument(
            "--exists",
            action="store_true",
            help="Indicates whether the folder structure already exists.",
        )



    def handler(self, context):
        name = (
            context.arguments.project_name
            or input("Project name [my_model]: ")
            or "my_model"
        )




        root = pathlib.Path(context.arguments.path) / name
        try:
            root.mkdir(exist_ok=context.arguments.exists)
        except FileExistsError:
            return report(
                f"Could not create '{root.absolute()}', directory exists.", level=0
            )
        ext = "json" if context.arguments.json else "yaml"

        plugin_paths = plugins.discover("config.templates")

        fileoptions = os.listdir(plugin_paths[ext + '_templates'][0])
        fileoptions = [file for file in fileoptions if ext in file]
        readline.set_completer(SimpleCompleter(fileoptions).complete)
        readline.parse_and_bind("tab: complete")

        if context.arguments.quickstart:
            template = f"starting_example.{ext}"
            output = f"network_configuration.{ext}"
        else:
            template = input(f"Config template [skeleton.{ext}]: ") or f"skeleton.{ext}"
            output = (
                input(f"Config filename [network_configuration.{ext}]: ")
                or f"network_configuration.{ext}"
            )
        config.copy_template(template, output=root / output)
        with open(root / "pyproject.toml", "w") as f:
            toml.dump(
                {
                    "tools": {
                        "bsb": {
                            "config": output,
                        }
                    }
                },
                f,
            )
        place_path = root / "placement.py"
        conn_path = root / "connectome.py"
        if not place_path.exists():
            with open(place_path, "w") as f:
                f.write("from bsb.placement import PlacementStrategy\n")
        if not conn_path.exists():
            with open(conn_path, "w") as f:
                f.write("from bsb.connectivity import ConnectionStrategy\n")

        report(f"Created '{name}' project structure.", level=1)
