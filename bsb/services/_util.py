import importlib

from ..exceptions import DependencyError


class ErrorModule:
    def __init__(self, message):
        self._msg = message

    def __getattr__(self, attr):
        raise DependencyError(self._msg)


class MockModule:
    def __new__(cls, module):
        try:
            instance = importlib.import_module(module)
        except ImportError:
            instance = super().__new__(cls)
            instance._mocked = True
        return instance
