from ..strategy import ConnectionStrategy
from importlib import import_module
from glob import glob
import os, inspect


# Scan the whole directory for python files, then import any ConnectionStrategies into
# this module.
src_files = glob(os.path.join(os.path.dirname(__file__), "*.py"))
exclude_src = ["__init__"]
for src_file in src_files:
    module_name = os.path.basename(src_file).split(".")[0]
    if module_name in exclude_src:
        continue
    module = import_module("." + module_name, __package__)
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, ConnectionStrategy):
            globals()[name] = obj
