import configparser
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
	help="Specify the path of the configuration .ini file.",
	default="mouse_cerebellum.ini"
)
cl_args = parser.parse_args()
pprint(cl_args)
