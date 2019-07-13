import configparser
import argparse
from scaffold.config import ScaffoldIniConfig

##
## This is the high-level API: It receives a command from the CLI, which is translated
## to the scaffold python package and executed.
##

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
	help="Specify the path of the configuration .ini file.",
	default="mouse_cerebellum.ini"
)
cl_args = parser.parse_args()
scaffoldConfig = ScaffoldIniConfig(cl_args.config)
print(scaffoldConfig.getLayer(id=scaffoldConfig.getLayerID('Molecular Layer')))
