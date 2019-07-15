import configparser
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
cl_args = parser.parse_args()


# Load the .ini configuration
scaffoldConfig = ScaffoldIniConfig(cl_args.config)

# Create the scaffold instance
scaffold = Scaffold(scaffoldConfig)
