__version__ = "4.0.0a30"

import functools

# Patch functools on 3.8
try:
    functools.cache
except AttributeError:
    functools.cache = functools.lru_cache

    # Patch the 'register' method of `singledispatchmethod` pre python 3.10
    def _register(self, cls, method=None):  # pragma: nocover
        if hasattr(cls, "__func__"):
            setattr(cls, "__annotations__", cls.__func__.__annotations__)
        return self.dispatcher.register(cls, func=method)

    functools.singledispatchmethod.register = _register


from .reporting import report, warn
