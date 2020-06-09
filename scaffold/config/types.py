from ..exceptions import *
import math


def any():
    def type_handler(value):
        return value

    type_handler.__name__ = "any"
    return type_handler


def in_(container):
    def type_handler(value):
        return value in container

    type_handler.__name__ = "any of the following values: " + str(_list(container))
    return type_handler


def or_(*type_args):
    handler_name = "any of: " + ", ".join(map(lambda x: x.__name__, type_args))

    def type_handler(value):
        for t in type_args:
            try:
                v = t(value)
                break
            except (TypeError, CastError):
                continue
        else:
            raise TypeError("Couldn't cast {} into {}".format(value, handler_name))
        return v

    type_handler.__name__ = handler_name
    return type_handler


def scalar_expand(scalar_type, size=None, expand=None):
    """
        Create a method that expands a scalar into an array with a specific size or uses
        an expansion function.

        :param scalar_type: Type of the scalar
        :type scalar_type: type
        :param size: Expand the scalar to an array of a fixed size.
        :type size: int
        :param expand: A function that takes the scalar value as argument and returns the expanded form.
        :type expand: callable
        :returns: Type handler function
        :rtype: callable
    """

    def type_handler(value):
        # No try block: let it raise the cast error.
        v = scalar_type(value)
        # Expand the scalar.
        return expand(v)

    type_handler.__name__ = "expanded list of " + scalar_type.__name__
    return type_handler


_list = list


def list(type=str, size=None):
    def type_handler(value):
        # Simple lists default to returning None for None, while configuration lists
        # default to an empty list.
        if value is None:
            return
        v = _list(value)
        try:
            for i, e in enumerate(v):
                v[i] = type(e)
        except:
            raise TypeError(
                "Couldn't cast element {} of {} into {}".format(i, value, type.__name__)
            )
        if size is not None and len(v) != size:
            raise ValueError(
                "Couldn't cast {} into a {} element list".format(value, size)
            )
        return v

    type_handler.__name__ = "list{} of {}".format(
        "[{}]".format(size) if size is not None else "", type.__name__
    )
    return type_handler


def fraction():
    def type_handler(value):
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError("{} is out of the 0-1 range for a fraction.".format(value))
        return v

    type_handler.__name__ = "fraction [0.; 1.]"
    return type_handler


def deg_to_radian():
    def type_handler(value):
        v = float(value)
        return float(v) * 2 * math.pi / 360

    type_handler.__name__ = "degrees"
    return type_handler
