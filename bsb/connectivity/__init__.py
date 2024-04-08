# isort: off
# Load module before others to prevent partially initialized modules
from .strategy import ConnectionStrategy

# isort: on
from .detailed import *
from .general import *
from .geometric import *
from .import_ import CsvImportConnectivity
