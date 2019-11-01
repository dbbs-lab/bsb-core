import sys
import configparser
import builtins
import argparse
from .config import JSONConfig, from_hdf5
from .scaffold import Scaffold
from .output import MorphologyRepository
import traceback

##
## This is the package entry point API and REPL.
##

def scaffold_cli():
	'''
		console_scripts entry point for the scaffold package. Will start the CLI handler or REPL handler.
	'''
	args = sys.argv[1:]
	if len(args) == 0: # Start the REPL?
		start_repl()
	else:
		start_cli()

def check_positive_factory(name):
	'''
		Return a function to report whether a certain value is a positive integer.
		If it isn't, raise an ArgumentTypeError.
	'''
	# Define factory product function.
	def f(x):
		try: # Try to cast the parameter to an int
			x = int(x)
			if x >= 0: # If x is positive, return it.
				return x
			# x is not positive, raise an exception.
			raise
		except Exception as e: # Catch the conversion or no-return exception and raise ArgumentTypeError.
			raise argparse.ArgumentTypeError("{} is an invalid {} value (positive int expected)".format(x, name))
	# Return factory product function.
	return f

def start_repl():
	'''
		Scaffold package REPL handler. Will parse user commands.
	'''
	# TODO: Switch to argparse and factory func based approach instead of string parsing
	# TODO: Add a python environment with access to scaffold instance
	# TODO: Port the debug voxel_cloud
	print("REPL not implemented yet. Use 'exit' to exit and use CLI command 'scaffold -h' for available CLI commands.")
	state = ReplState()
	while not state.exit:
		state.repl()

def start_cli():
	'''
		Scaffold package CLI handler
	'''
	# Parser
	parser = argparse.ArgumentParser()
	# Subparsers
	subparsers = parser.add_subparsers(
		title='Scaffold tasks',
		description='The scaffold performs multiple seperate tasks. See the list below for available tasks.',
		dest='task'
	)
	parser_compile = subparsers.add_parser('compile', help='Build a network of neurons in a volume and compile it to an HDF5 network architecture file.')
	parser_run = subparsers.add_parser('run', help='Run a simulation from scratch.')
	parser_sim = subparsers.add_parser('simulate', help='Run a simulation from a compiled HDF5 network architecture file.')
	parser_repl = subparsers.add_parser('repl')

	# Main arguments
	parser.add_argument("-c", "--config",
		help="Specify the path of the configuration file.",
		default="{}/configurations/mouse_cerebellum.json".format(sys.modules[globals()['__package__']].__path__[0])
	)
	parser.add_argument("-ct", "--ctype",
		help="Specify the type of the configuration file.",
		default="json", choices=["json"]
	)
	parser.add_argument("-v", "--verbose",
		help="Specify the verbosity of the console output",
		default=1, type=check_positive_factory("verbosity")
	)
	parser.add_argument("-o", "--output", help="Specify an output file path")

	# Compile subparser
	parser_compile.add_argument('-p', action='store_true',help='Plot the created network')

	# Run subparser
	parser_run.add_argument('simulation', action='store', help='Preconfigured simulation to run.')
	parser_run.add_argument('-p', action='store_true',help='Plot the created network')

	# Simulate subparser
	parser_sim.add_argument('simulation', action='store', help='Name of the preconfigured simulation.')
	parser_sim.add_argument('--hdf5', action='store', required=True, help='Name of the HDF5 file to load.')

	# Repl subparser
	parser_repl.set_defaults(func=start_repl)

	cl_args = parser.parse_args()
	if hasattr(cl_args, 'func'):
		cl_args.func(cl_args)
	else:
		file = None
		if hasattr(cl_args, 'hdf5'): # Is an HDF5 file specified?
			cl_args.ctype = 'hdf5' # Load from the config stored in the HDF5 file.

		if cl_args.ctype == 'json': # Should we config from JSON?
			# Load the .json configuration
			scaffoldConfig = JSONConfig(file=cl_args.config, verbosity=cl_args.verbose)
		elif cl_args.ctype == 'hdf5': # Should we config from hdf5?
			file = cl_args.hdf5
			scaffoldConfig = from_hdf5(file) # Extract the config stored in the hdf5 file.

		# Create the scaffold instance
		scaffoldInstance = Scaffold(scaffoldConfig, from_file=file) # `from_file` notifies the scaffold instance that we might've loaded from a file.

		if cl_args.output: # Is a new output file name specified?
			scaffoldInstance.output_formatter.save_file_as = cl_args.output

		if cl_args.task == 'compile' or cl_args.task == 'run': # Do we need to compile a network architecture?
			scaffoldInstance.compile_network()
			if cl_args.p: # Is a plot requested?
				scaffoldInstance.plot_network_cache()

		if cl_args.task == 'run' or cl_args.task == 'simulate': # Do we need to run a simulation?
			scaffoldInstance.run_simulation(cl_args.simulation)

class ReplState:
	'''
		Stores the REPL state and executes each step of the REPL.
	'''
	def __init__(self):
		self.exit = False
		self.state = 'base'
		self.question = None
		self.prefix = None
		self.reply = None
		self.next = None
		self.command = None
		self.error = False
		self.globals = {}

	def repl(self):
		'''
			Execute the next repl step.
		'''
		self.command = input('{}{}{}> '.format(
			(self.reply or '') + ('\n' if self.reply else ''),
			(self.question or '') + ('\n' if self.question else ''),
			self.prefix or ''
		))
		if self.command == "":
			return
		self.exit = self.command == "exit"
		self.state = self.next or self.state
		self.next = None
		self.question = None
		self.reply = None
		self.update_parser()

		try:
			args = self.parser.parse_args(self.command.split(" "))
		except ParseException as e:
			print(str(e))
			return

		if hasattr(args, "func"):
			try:
				args.func(args)
			except Exception as e:
				traceback.print_exc()

	def update_parser(self):
		self.parser = StateParser(add_help=False)
		self.subparsers = self.parser.add_subparsers()
		self.add_parser_globals()
		state_method_name = 'set_parser_{}_state'.format(self.state)
		state_method = getattr(self.__class__, state_method_name, None)
		if callable(state_method):
			state_method(self)
		else:
			raise Exception("Unparsable state: {}".format(self.state))


	def set_parser_base_state(self):
		parser_open = self.add_subparser("open", description="Open various scaffold files like output files, repositories and tree collections.")
		open_subparsers = parser_open.add_subparsers()

		mr_parser = open_subparsers.add_parser("mr", description="Open a morphology repository.")
		mr_parser.add_argument('file', action='store', help='Path of the morphology repository to load.')
		mr_parser.set_defaults(func=self.open_morphology_repository)


	def set_parser_base_mr_state(self):
		mr = self.globals["mr"]

		close_parser = self.add_subparser("close", description="Close the currently opened repository.")
		close_parser.set_defaults(func=lambda args: (self.clear_prefix() and False) or self.set_next_state("base"))

		list_parser = self.add_subparser("list")
		list_subparsers = list_parser.add_subparsers()

		all_parser = list_subparsers.add_parser("all", description="List all morphologies in the repository")
		all_parser.set_defaults(func=
			lambda args: self.set_reply(self.globals["mr"].list_all_morphologies())
		)

		voxelized_parser = list_subparsers.add_parser("voxelized", description="List voxelized morphologies in the repository")
		voxelized_parser.set_defaults(func=
			lambda args: self.set_reply(self.globals["mr"].list_all_voxelized())
		)

		import_parser = self.add_subparser("import", description="Import a morphology or repository into the repository")
		import_parser.add_argument("file", action="store", help="Filename of the swc file.")
		import_parser.add_argument("name", action="store", help="Unique name of the morphology.")
		import_parser.set_defaults(func=
			lambda args: (mr.import_swc(args.file, args.name, overwrite=True) and False)
			or self.set_reply( "Added '{}' as '{}' to the repository.".format(args.file, args.name))
		)

	def add_parser_globals(self):
		exit_parser = self.add_subparser("exit")
		exit_parser.set_defaults(func=self.exit_repl)

	def add_subparser(self, *args, **kwargs):
		return self.subparsers.add_parser(*args, **kwargs)

	def exit_repl(self, args):
		exit()

	def clear_prefix(self):
		self.prefix = None

	def set_next_state(self, state):
		self.next = state

	def set_reply(self, message):
		self.reply = str(message)

	def open_morphology_repository(self, args):
		mr = MorphologyRepository(args.file)
		self.globals["mr"] = mr
		self.next = "base_mr"
		self.prefix = "repo <'{}'".format(args.file)

class ParseException(Exception):
	pass

class StateParser(argparse.ArgumentParser):

	def error(self, message):
		raise ParseException(message)
