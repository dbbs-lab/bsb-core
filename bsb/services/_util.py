from ..exceptions import DependencyError as _DepErr
import importlib


class ErrorModule:
    def __init__(self, message):
        self._msg = message

    def __getattr__(self, attr):
        raise _DepErr(self._msg)


class MockModule:
    def __new__(cls, module):
        try:
            instance = importlib.import_module(module)
        except ImportError:
            instance = super().__new__(cls)
            instance._mocked = True
        return instance


def mock_error(f):
    def wrapper(*args, **kwargs):
        print(f.__closure__)
        raise _DepErr(f)
