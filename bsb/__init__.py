"""
`bsb-core` is the backbone package contain the essential code of the BSB: A component
framework for multiscale bottom-up neural modelling.

`bsb-core` needs to be installed alongside a bundle of desired bsb plugins, some of
which are essential for `bsb-core` to function. First time users are recommended to
install the `bsb` package instead.
"""

__version__ = "4.0.0b9"

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

try:
    from .options import profiling as _pr
except Exception:
    _pr = False

if _pr:
    from .profiling import activate_session

    session = activate_session()
    meter = session.meter("root_module")
    meter.start()

from . import reporting

reporting.setup_reporting()

if _pr:
    meter.stop()
