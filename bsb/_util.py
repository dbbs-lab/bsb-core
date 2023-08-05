import itertools as _it
import abc as _abc
import os as _os
import sys as _sys
import contextlib as _ctxlib
import typing
import numpy as np

import numpy as _np
from .exceptions import OrderError as _OrderError
import functools

ichain = _it.chain.from_iterable


def merge_dicts(a, b):
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = b[key]
    return a


def obj_str_insert(__str__):
    @functools.wraps(__str__)
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


class SortableByAfter:
    @_abc.abstractmethod
    def has_after(self):
        pass

    @_abc.abstractmethod
    def create_after(self):
        pass

    @_abc.abstractmethod
    def get_after(self):
        pass

    @_abc.abstractmethod
    def get_ordered(self, objects):
        pass

    def add_after(self, after_item):
        if not self.has_after():
            self.create_after()
        self.get_after().append(after_item)

    def is_after_satisfied(self, objects):
        """
        Determine whether the ``after`` specification of this object is met. Any objects
        appearing in ``self.after`` need to occur in ``objects`` before the object.

        :param objects: Proposed order for which the after condition is checked.
        :type objects: list
        """
        if not self.has_after():  # No after?
            # Condition without constraints always True.
            return True
        self_met = False
        after = self.get_after()
        # Determine whether this object is out of order.
        for type in objects:
            if type is self:
                # We found ourselves, from this point on nothing that appears in the after
                # array is allowed to be encountered
                self_met = True
            elif self_met and type in after:
                # We have encountered ourselves, so everything we find from now on is not
                # allowed to be in our after array, if it is, we fail the after condition.
                return False
        # We didn't meet anything behind us that was supposed to be in front of us
        # => Condition met.
        return True

    def satisfy_after(self, objects):
        """
        Given an array of objects, place this object after all of the objects specified in
        the ``after`` condition. If objects in the after condition are missing from the
        given array this object is placed at the end of the array. Modifies the `objects`
        array in place.
        """
        before_types = self.get_after().copy()
        i = 0
        place_after = False
        # Loop over the objects until we've found all our before types.
        while len(before_types) > 0 and i < len(objects):
            if objects[i] in before_types:
                # We encountered one of our before types and can remove it from the list
                # of things we still need to look for
                before_types.remove(objects[i])
            # We increment i unless we encounter and remove ourselves
            if objects[i] == self:
                # We are still in the loop, so there must still be things in our after
                # condition that we are looking for; therefor we remove ourselves from
                # the list and wait until we found all our conditions and place ourselves
                # there
                objects.remove(self)
                place_after = True
            else:
                i += 1
        if place_after:
            # We've looped to either after our last after condition or the last element
            # and should reinsert ourselves here.
            objects.insert(i, self)

    @classmethod
    def resolve_order(cls, objects):
        """
        Orders a given dictionary of objects by the class's default mechanism and
        then apply the `after` attribute for further restrictions.
        """
        # Sort by the default approach
        sorting_objects = list(cls.get_ordered(objects))
        # Afterwards cell types can be specified that need to be placed after other types.
        after_specifications = [c for c in sorting_objects if c.has_after()]
        j = 0
        # Keep rearranging as long as any cell type's after condition isn't satisfied.
        while any(
            not c.is_after_satisfied(sorting_objects) for c in after_specifications
        ):
            j += 1
            # Rearrange each element that is out of place.
            for after_type in after_specifications:
                if not after_type.is_after_satisfied(sorting_objects):
                    after_type.satisfy_after(sorting_objects)
            # If we have had to rearrange all elements more than there are elements, the
            # conditions cannot be met, and a circular dependency is at play.
            if j > len(objects):
                circulars = ", ".join(
                    c.name
                    for c in after_specifications
                    if not c.is_after_satisfied(sorting_objects)
                )
                raise _OrderError(
                    f"Couldn't resolve order, probably a circular dependency including: "
                    f"{circulars}"
                )
        # Return the sorted array.
        return sorting_objects


def immutable():
    def immutable_decorator(f):
        @functools.wraps(f)
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
