__version__ = "4.0.0a8"

import functools

# Patch functools on 3.8
try:
    functools.cache
except AttributeError:
    functools.cache = functools.lru_cache

from ._mpi import *
from .reporting import report, warn
