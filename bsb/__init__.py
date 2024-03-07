"""
`bsb-core` is the backbone package contain the essential code of the BSB: A component
framework for multiscale bottom-up neural modelling.

`bsb-core` needs to be installed alongside a bundle of desired bsb plugins, some of
which are essential for `bsb-core` to function. First time users are recommended to
install the `bsb` package instead.
"""

__version__ = "4.0.0b9"

import ast
import functools
import importlib
import sys
from pathlib import Path

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


@functools.cache
def _get_public_api_map():
    root = Path(__file__).parent

    public_api_map = {}
    for file in root.rglob("*.py"):
        module_parts = file.relative_to(root).parts
        module = ".".join(
            module_parts[:-1]
            + ((module_parts[-1][:-3],) if module_parts[-1] != "__init__.py" else tuple())
        )
        module_api = []
        for assign in ast.parse(file.read_text()).body:
            if isinstance(assign, ast.Assign) and any(
                target.id == "__all__"
                for target in assign.targets
                if isinstance(target, ast.Name)
            ):
                if isinstance(assign.value, ast.List):
                    module_api = [
                        el.value
                        for el in assign.value.elts
                        if isinstance(el, ast.Constant)
                    ]
        for api in module_api:
            public_api_map[api] = module

    return public_api_map


@functools.cache
def __getattr__(name):
    if name == "config":
        return object.__getattribute__(sys.modules[__name__], name)
    api = _get_public_api_map()
    module = api.get(name, None)
    if module is None:
        return object.__getattribute__(sys.modules[__name__], name)
    else:
        return getattr(importlib.import_module("." + module, package="bsb"), name)
