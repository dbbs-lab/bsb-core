import configparser
import builtins
import argparse
from scaffold.config import ScaffoldIniConfig
from scaffold.scaffold import Scaffold

##
## This is the high-level API: It receives a command from the CLI, which is translated
## to the scaffold python package and executed.
##

# Parse the command line arguments.

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
help="Specify the path of the configuration .ini file.",
default="mouse_cerebellum.ini"
)
subparsers = parser.add_subparsers(
	title='Scaffold tasks',
	description='The scaffold performs multiple seperate tasks. See the list below for available tasks.',
	required=True,
	dest='task'
)

parser_compile = subparsers.add_parser('compile', help='Build a network of neurons in a volume.')

parser_run = subparsers.add_parser('run', help='Run a simulation using a compiled scaffold network.')

cl_args = parser.parse_args()

# Load the .ini configuration
scaffoldConfig = ScaffoldIniConfig(cl_args.config)

# Create the scaffold instance
scaffoldInstance = Scaffold(scaffoldConfig)
# Make the scaffoldInstance available in all modules. (Monkey patch during rework)
builtins.scaffoldInstance = scaffoldInstance

if cl_args.task == 'compile':
	# Use the configuration to initialise all components such as cells and layers
	# to prepare for the network architecture compilation.
	scaffoldInstance.initialiseComponents()
	# Run the procedural file network_architecture.py
	from network_architecture import *

if cl_args.task == 'run':
	# Run the nest script
	pass
