import sys
import configparser
import builtins
import argparse
from .config import JSONConfig, from_hdf5
from .scaffold import Scaffold

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
	state = ReplState()
	while not state.exit:
		print("REPL not implemented yet. Use 'exit' to exit and use CLI command 'scaffold -h' for available CLI commands.")
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

	def repl(self):
		'''
			Execute the next repl step.
		'''
		self.command = input('{}{}{}> '.format(
			(self.reply or '') + ('\n' if self.reply else ''),
			(self.question or '') + ('\n' if self.question else ''),
			self.prefix or ''
		))
		self.exit = self.command == "exit"
		self.state = self.next or self.state
		self.next = None
		self.question = None
		self.prefix = None
		self.reply = None
