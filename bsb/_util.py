import contextlib as _ctxlib
import functools as _ft
import inspect as _inspect
import itertools as _it
import os as _os
import sys as _sys
import typing

import numpy as _np
import numpy as np

ichain = _it.chain.from_iterable


def merge_dicts(a, b):
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def obj_str_insert(__str__):
    @_ft.wraps(__str__)
    def wrapper(self):
        obj_str = object.__repr__(self)
        return obj_str.replace("at 0x", f"{__str__(self)} at 0x")

    return wrapper


@_ctxlib.contextmanager
def suppress_stdout():
    with open(_os.devnull, "w") as devnull:
        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        _sys.stdout = devnull
        _sys.stderr = devnull
        try:
            yield
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr


def get_qualified_class_name(x):
    """Return an object's module and class name"""
    if _inspect.isclass(x):
        return f"{x.__module__}.{str(x.__name__)}"
    return f"{x.__class__.__module__}.{str(x.__class__.__name__)}"


def listify_input(value):
    """
    Turn any non-list values into a list containing the value. Sequences will be
    converted to a list using `list()`, `None` will  be replaced by an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [str]
    try:
        return list(value)
    except Exception:
        return [value]


def sanitize_ndarray(arr_input, shape, dtype=None):
    kwargs = {"copy": False}
    if dtype is not None:
        kwargs["dtype"] = dtype
    arr = _np.array(arr_input, **kwargs)
    arr.shape = shape
    return arr


def assert_samelen(*args):
    len_ = None
    assert all(
        (len_ := len(arg) if len_ is None else len(arg)) == len_ for arg in args
    ), "Input arguments should be of same length."


def immutable():
    def immutable_decorator(f):
        @_ft.wraps(f)
        def immutable_action(self, *args, **kwargs):
            new_instance = self.__copy__()
            f(new_instance, *args, **kwargs)
            return new_instance

        return immutable_action

    return immutable_decorator


def unique(iter_: typing.Iterable[typing.Any]):
    return [*set(iter_)]


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2

    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if (
        np.isnan(vec1).any()
        or np.isnan(vec2).any()
        or not np.any(vec1)
        or not np.any(vec2)
    ):
        raise ValueError("Vectors should not contain nan and their norm should not be 0.")
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions
