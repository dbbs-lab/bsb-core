__version__ = "3.10.4"

import functools

# Patch functools on 3.8
try:
    functools.cache
except AttributeError:
    functools.cache = functools.lru_cache

from .reporting import set_verbosity, report, warn
