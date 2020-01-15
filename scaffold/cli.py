"""
    This module contains all classes and functions required to run the scaffold
    from the command line.
"""

import sys
import configparser
import builtins
import argparse
import traceback

##
## This is the package entry point API and REPL.
##


def scaffold_cli():
    """
        console_scripts entry point for the scaffold package. Will start the CLI handler or REPL handler.
    """
    args = sys.argv[1:]
    if len(args) == 0:  # Start the REPL?
        start_repl()
    else:
        start_cli()


def check_positive_factory(name):
    """
        Return a function to report whether a certain value is a positive integer.
        If it isn't, raise an ArgumentTypeError.
    """
    # Define factory product function.
    def f(x):
        try:  # Try to cast the parameter to an int
            x = int(x)
            if x >= 0:  # If x is positive, return it.
                return x
            # x is not positive, raise an exception.
            raise
        except Exception as e:  # Catch the conversion or no-return exception and raise ArgumentTypeError.
            raise argparse.ArgumentTypeError(
                "{} is an invalid {} value (positive int expected)".format(x, name)
            )

    # Return factory product function.
    return f


def start_repl():
    """
        Scaffold package REPL handler. Will parse user commands.
    """
    # TODO: Add a python environment with access to globals like a scaffold or morphology repository
    state = ReplState()
    try:
        while not state.exit:
            state.repl()
    finally:
        state.destroy_globals()


def start_cli():
    """
        Scaffold package CLI handler
    """
    # Parser
    parser = argparse.ArgumentParser()
    # Subparsers
    subparsers = parser.add_subparsers(
        title="Scaffold tasks",
        description="The scaffold performs multiple seperate tasks. See the list below for available tasks.",
        dest="task",
    )
    parser_compile = subparsers.add_parser(
        "compile",
        help="Build a network of neurons in a volume and compile it to an HDF5 network architecture file.",
    )
    parser_run = subparsers.add_parser("run", help="Run a simulation from scratch.")
    parser_sim = subparsers.add_parser(
        "simulate",
        help="Run a simulation from a compiled HDF5 network architecture file.",
    )
    parser_config = subparsers.add_parser(
        "make-config", help="Create a config file in the current directory."
    )
    parser_repl = subparsers.add_parser(
        "repl", help="Start the interactive scaffold shell."
    )
    parser_plot = subparsers.add_parser("plot", help="Plot networks.")

    # Main arguments
    parser.add_argument(
        "-c",
        "--config",
        help="Specify the path of the configuration file.",
        default="mouse_cerebellum.json",
    )
    parser.add_argument(
        "-ct",
        "--ctype",
        help="Specify the type of the configuration file.",
        default="json",
        choices=["json"],
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Specify the verbosity of the console output",
        default=1,
        type=check_positive_factory("verbosity"),
    )
    parser.add_argument("-o", "--output", help="Specify an output file path")

    # Compile subparser
    parser_compile.add_argument(
        "-p", action="store_true", help="Plot the created network"
    )
    parser_compile.add_argument("-x", help="Resize volume X")
    parser_compile.add_argument("-z", help="Resize volume Z")

    # Run subparser
    parser_run.add_argument(
        "simulation", action="store", help="Preconfigured simulation to run."
    )
    parser_run.add_argument("-p", action="store_true", help="Plot the created network")
    parser_run.add_argument("-o", "--output", help="Specify an output file path")
    parser_run.add_argument(
        "-rc", "--reconfigure", help="Specify the path of the new configuration file."
    )
    parser_run.add_argument("-x", help="Resize volume X")
    parser_run.add_argument("-z", help="Resize volume Z")

    # Simulate subparser
    parser_sim.add_argument(
        "simulation", action="store", help="Name of the preconfigured simulation."
    )
    parser_sim.add_argument(
        "--hdf5", action="store", required=True, help="Name of the HDF5 file to load."
    )
    parser_sim.add_argument(
        "-rc", "--reconfigure", help="Specify the path of the new configuration file."
    )

    # Create config subparser
    parser_config.add_argument(
        "-t",
        "--template",
        action="store",
        default="mouse_cerebellum.json",
        help="Name of the template config file.",
    )
    parser_config.add_argument(
        "output",
        action="store",
        default="scaffold_configuration.json",
        nargs="?",
        help="Name of the output configuration file.",
    )
    parser_config.set_defaults(func=create_config)

    parser_plot.add_argument("hdf5", action="store", help="Path of the HDF5 file")
    parser_plot.set_defaults(func=cli_plot)

    # Repl subparser
    parser_repl.set_defaults(func=start_repl)

    cl_args = parser.parse_args()
    if hasattr(cl_args, "func"):
        cl_args.func(cl_args)
    else:
        from .config import JSONConfig, _from_hdf5
        from .scaffold import Scaffold
        from .output import MorphologyRepository, HDF5Formatter

        file = None
        if hasattr(cl_args, "hdf5"):  # Is an HDF5 file specified?
            cl_args.ctype = "hdf5"  # Load from the config stored in the HDF5 file.

        if cl_args.ctype == "json":  # Should we config from JSON?
            # Load the .json configuration
            scaffoldConfig = JSONConfig(file=cl_args.config, verbosity=cl_args.verbose)
        elif cl_args.ctype == "hdf5":  # Should we config from hdf5?
            file = cl_args.hdf5
            if cl_args.reconfigure is not None:
                print(cl_args.reconfigure)
                config = JSONConfig(file=cl_args.reconfigure)
                HDF5Formatter.reconfigure(file, config)
            scaffoldConfig = _from_hdf5(
                file, verbosity=cl_args.verbose
            )  # Extract the config stored in the hdf5 file.

        # Create the scaffold instance
        scaffoldInstance = Scaffold(
            scaffoldConfig, from_file=file
        )  # `from_file` notifies the scaffold instance that we might've loaded from a file.

        if hasattr(cl_args, "x") and hasattr(cl_args, "z") and (cl_args.x or cl_args.z):
            kwargs = {"X": float(cl_args.x), "Z": float(cl_args.z)}
            print("resizing:", kwargs)
            scaffoldInstance.configuration.resize(**kwargs)

        if cl_args.output:  # Is a new output file name specified?
            scaffoldInstance.output_formatter.save_file_as = cl_args.output

        if (
            cl_args.task == "compile" or cl_args.task == "run"
        ):  # Do we need to compile a network architecture?
            scaffoldInstance.compile_network()
            if cl_args.p:  # Is a plot requested?
                scaffoldInstance.plot_network_cache()

        if (
            cl_args.task == "run" or cl_args.task == "simulate"
        ):  # Do we need to run a simulation?
            scaffoldInstance.run_simulation(cl_args.simulation)


def cli_plot(args):
    from .scaffold import from_hdf5
    from .plotting import plot_network

    scaffold = from_hdf5(args.hdf5)
    plot_network(scaffold, from_memory=False)


def create_config(args):
    from .helpers import get_config_path
    from shutil import copy2 as copy_file
    import os

    copy_file(get_config_path(args.template), "./" + args.output)


class ReplState:
    """
        Stores the REPL state and executes each step of the REPL.
    """

    def __init__(self):
        self.exit = False
        self.state = "base"
        self.question = None
        self.prefix = None
        self.reply = None
        self.next = None
        self.command = None
        self.error = False
        self.globals = {}

    def repl(self):
        """
            Execute the next repl step.
        """
        # Ask for user input
        self.command = input(
            "{}{}{}> ".format(
                (self.reply or "") + ("\n" if self.reply else ""),
                (self.question or "") + ("\n" if self.question else ""),
                self.prefix or "",
            )
        )
        if self.command == "":  # Empty command? Next step.
            return
        # Update state with next if set, or keep last state.
        self.state = self.next or self.state
        # Reset transient values
        self.next = None
        self.question = None
        self.reply = None
        # Add subparsers and arguments based on the state
        self.update_parser()

        try:
            # Parse the command
            args = self.parser.parse_args(self.command.split(" "))
        except ParseException as e:
            print(str(e))
            return

        if hasattr(args, "func"):  # Was a command found?
            try:
                # Execute the command
                args.func(args)
            except Exception as e:
                # Catch and print all exceptions
                traceback.print_exc()

    def update_parser(self):
        """
            Creates a new parser for the next REPL step. Tries to add
            subparsers and arguments if the method "set_parser_``state``_state"
            is callable.
        """
        self.parser = StateParser(add_help=False)
        self.subparsers = self.parser.add_subparsers()
        self.add_parser_globals()
        state_method_name = "set_parser_{}_state".format(self.state)
        state_method = getattr(self.__class__, state_method_name, None)
        if callable(state_method):
            state_method(self)
        else:
            raise Exception("Unparsable state: {}".format(self.state))

    def set_parser_base_state(self):
        """
            Adds the initial subparsers and arguments to the REPL parser.
        """
        parser_open = self.add_subparser(
            "open",
            description="Open various scaffold files like output files, repositories and tree collections.",
        )
        open_subparsers = parser_open.add_subparsers()

        mr_parser = open_subparsers.add_parser(
            "mr", description="Open a morphology repository."
        )
        mr_parser.add_argument(
            "file", action="store", help="Path of the morphology repository to load."
        )
        mr_parser.set_defaults(func=self.open_morphology_repository)

        hdf5_parser = open_subparsers.add_parser("hdf5", description="Open a HDF5 file.")
        hdf5_parser.add_argument(
            "file", action="store", help="Path of the HDF5 file to load."
        )
        hdf5_parser.set_defaults(func=self.open_hdf5)

    def set_parser_base_mr_state(self):
        """
            Adds the morphology repository state subparsers and arguments to
            the REPL parser.
        """
        mr = self.globals["mr"]

        close_parser = self.add_subparser(
            "close", description="Close the currently opened repository."
        )
        close_parser.set_defaults(
            func=lambda args: (self.clear_prefix() and False)
            or self.set_next_state("base")
        )

        list_parser = self.add_subparser("list")
        list_subparsers = list_parser.add_subparsers()

        all_parser = list_subparsers.add_parser(
            "all", description="List all morphologies in the repository"
        )
        all_parser.set_defaults(
            func=lambda args: self.set_reply(self.globals["mr"].list_all_morphologies())
        )

        voxelized_parser = list_subparsers.add_parser(
            "voxelized", description="List voxelized morphologies in the repository"
        )
        voxelized_parser.set_defaults(
            func=lambda args: self.set_reply(self.globals["mr"].list_all_voxelized())
        )

        import_parser = self.add_subparser(
            "import", description="Import a morphology or repository into the repository"
        )
        types_subparsers = import_parser.add_subparsers()

        swc_parser = types_subparsers.add_parser("swc", description="Import SWC file")
        swc_parser.add_argument("file", action="store", help="Filename of the swc file.")
        swc_parser.add_argument(
            "name", action="store", help="Unique name of the morphology."
        )
        swc_parser.set_defaults(
            func=lambda args: mr.import_swc(args.file, args.name, overwrite=True)
        )

        remove_parser = self.add_subparser(
            "remove", description="Remove a morphology from the repository.."
        )
        remove_parser.add_argument("name", help="Name of the morphology")
        remove_parser.set_defaults(func=lambda args: mr.remove_morphology(args.name))

        repo_parser = types_subparsers.add_parser(
            "repo", description="Import a morphology repository"
        )
        repo_parser.add_argument(
            "file", action="store", help="Filename of the repository."
        )
        repo_parser.add_argument(
            "-f", "--overwrite", action="store_true", dest="overwrite", default=False
        )
        repo_parser.set_defaults(
            func=lambda args: mr.import_repository(
                MorphologyRepository(args.file), overwrite=args.overwrite
            )
        )

        voxelize_parser = self.add_subparser(
            "voxelize", description="Divide a morphology into a given amount of voxels."
        )
        voxelize_parser.add_argument("name", help="Name of the morphology")
        voxelize_parser.add_argument("voxels", nargs="?", default=130, type=int)
        voxelize_parser.set_defaults(func=lambda args: repl_voxelize(mr, args))

        plot_parser = self.add_subparser("plot", description="Plot a morphology.")
        plot_parser.add_argument("name", help="Name of the morphology")
        plot_parser.set_defaults(func=lambda args: repl_plot_morphology(mr, args))

    def set_parser_base_hdf5_state(self):
        """
            Adds the HDF5 state subparsers and arguments to the REPL parser.
        """
        h = self.globals["hdf5"]

        def close(args):
            self.clear_prefix()
            self.close_hdf5()
            self.set_next_state("base")

        close_parser = self.add_subparser(
            "close", description="Close the currently opened HDF5 file."
        )
        close_parser.set_defaults(func=lambda args: close)
        view_parser = self.add_subparser(
            "view", description="Explore the hierarchical components of the HDF5 file."
        )
        view_parser.set_defaults(func=lambda args: repl_view_hdf5(h, args))

        plot_parser = self.add_subparser("plot", description="Plot the HDF5 network.")

        def plot_handler(args):
            args.hdf5 = h.filename
            cli_plot(args)

        plot_parser.set_defaults(func=plot_handler)

    def add_parser_globals(self):
        """
            Adds subparsers and arguments that should be there in any state.
        """
        exit_parser = self.add_subparser("exit")
        exit_parser.set_defaults(func=self.exit_repl)

    def add_subparser(self, *args, **kwargs):
        """
            Add a top level subparser to the current REPL parser.
        """
        return self.subparsers.add_parser(*args, **kwargs)

    def exit_repl(self, args):
        """
            Exit the REPL.
        """
        self.exit = True

    def clear_prefix(self):
        """
            Clear the REPL prefix.
        """
        self.prefix = None

    def set_next_state(self, state):
        """
            Set the next REPL state.

            :param state: The next state. For each state there should be a set_parser_``state``_state function (e.g. :func:`set_parser_base_state`).
            :type state: string
            :rtype: None
        """
        self.next = state

    def set_reply(self, message):
        """
            Set the REPL reply, to be printed to the user at the end of this step.

            :param message: The reply to print.
            :type message: string
            :rtype: None
        """
        self.reply = str(message)

    def close_hdf5(self):
        """
            Closes the currently open HDF5 file.

            :raises ParseException: Raised if there's no open HDF5 file.
            :rtype: None
        """
        if self.globals["hdf5"] is None:
            raise ParseException("No HDF5 file is currently opened.")
        self.globals["hdf5"].close()
        self.globals["hdf5"] = None

    def destroy_globals(self):
        """
            Always called before the REPL exits to clean up open resources.
        """
        if "hdf5" in self.globals and not self.globals["hdf5"] is None:
            self.close_hdf5()

    def open_morphology_repository(self, args):
        """
            Callback function that handles the ``open mr`` command.

            :param args: Result of ArgumentParser.parse_args()
            :type args: Namespace
            :rtype: None
        """
        # Create the morphology repository instance.
        mr = MorphologyRepository(args.file)
        # Store it as a global REPL variable.
        self.globals["mr"] = mr
        # Switch to the base_mr state.
        self.next = "base_mr"
        # Indicate the open mr state with a prefix.
        self.prefix = "repo <'{}'".format(args.file)

    def open_hdf5(self, args):
        """
            Callback function that handles the ``open hdf5`` command.

            :param args: Result of ArgumentParser.parse_args()
            :type args: Namespace
            :rtype: None
        """
        # Import the HDF5 library.
        import h5py

        # Add the h5py file handle as a global REPL variable.
        self.globals["hdf5"] = h5py.File(args.file, "a")
        # Switch to the base_hdf5 state.
        self.next = "base_hdf5"
        # Indicate the open hdf5 state with a prefix
        self.prefix = "hdf5 <'{}'".format(args.file)


class ParseException(Exception):
    """
        Thrown when the parsing of a command string fails.
    """

    pass


class StateParser(argparse.ArgumentParser):
    """
        Inherits from argparse.ArgumentParser and overloads the ``error``
        method so that when an error occurs, instead of exiting and exception
        is thrown.
    """

    def error(self, message):
        """
            Overloads default exit behavior with throwing ParseException.
        """
        raise ParseException(message)


def repl_plot_morphology(morphology_repository, args):
    """
        Callback function that handles ``plot`` command in the *base_mr* state.
    """
    m = morphology_repository.get_morphology(args.name)
    from .plotting import plot_morphology

    plot_morphology(m)


def repl_voxelize(morphology_repository, args):
    """
        Callback function that handles ``voxelize`` command in the *base_mr* state.
    """
    m = morphology_repository.get_morphology(args.name)
    m.voxelize(args.voxels)
    morphology_repository.store_voxel_cloud(m, overwrite=True)


def repl_view_hdf5(handle, args):
    """
        Callback function that handles ``view`` command in the *base_hdf5* state.
    """
    df = chr(172)

    def format_level(lvl, sub=None):
        return " " * (lvl * 3) + (sub or df)

    def format_self(obj, name, lvl):
        print(format_level(lvl) + name)
        if hasattr(obj, "attrs"):
            for attr in obj.attrs.keys():
                print(
                    " "
                    + format_level(lvl, ".")
                    + attr
                    + " = "
                    + str(obj.attrs[attr])[: min(100, len(str(obj.attrs[attr])))]
                    + ("..." if len(str(obj.attrs[attr])) > 100 else "")
                )
        if hasattr(obj, "keys"):
            for key in obj.keys():
                format_self(obj[key], obj[key].name, lvl + 1)

    format_self(handle, str(handle.file), 0)
