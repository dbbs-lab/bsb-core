from . import BaseCommand
from ...option import BsbOption
from ...reporting import report
from ... import config
import pathlib
import toml


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

        (root / name).mkdir(exist_ok=context.arguments.exists)
        if context.arguments.quickstart:
            template = "starting_example.json"
            output = "network_configuration.json"
        else:
            template = input("Config template [skeleton.json]: ") or "skeleton.json"
            output = (
                input("Config filename [network_configuration.json]: ")
                or "network_configuration.json"
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
