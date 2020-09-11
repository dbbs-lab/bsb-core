from ..exceptions import *
import math, sys


def any():
    def type_handler(value):
        return value

    type_handler.__name__ = "any"
    return type_handler


def in_(container):
    """
        Type validator. Checks whether the given value occurs in the given container.
        Uses the `in` operator.

        :param container: List of possible values
        :type container: container
        :returns: Type validator function
        :rtype: function
    """
    error_msg = "a value in: " + str(container)

    def type_handler(value):
        if value in container:
            return value
        else:
            raise TypeError("Couldn't cast '{}' to ".format(value) + error_msg)

    type_handler.__name__ = error_msg
    return type_handler


def or_(*type_args):
    """
        Type validator. Attempts to cast the value to any of the given types in order.

        :param type_args: Another type validator
        :type type_args: function
        :returns: Type validator function
        :raises: TypeError if none of the given type validators can cast the value.
        :rtype: function
    """
    handler_name = "any of: " + ", ".join(map(lambda x: x.__name__, type_args))

    def type_handler(value):
        for t in type_args:
            try:
                v = t(value)
                break
            except (TypeError, ValueError, CastError):
                continue
        else:
            raise TypeError("Couldn't cast {} into {}".format(value, handler_name))
        return v

    type_handler.__name__ = handler_name
    return type_handler


_int = int


def int(min=None, max=None):
    """
        Type validator. Attempts to cast the value to an int, optionally within some
        bounds.

        :param min: Minimum valid value
        :type min: int
        :param max: Maximum valid value
        :type max: int
        :returns: Type validator function
        :raises: TypeError when value can't be cast.
        :rtype: function
    """
    handler_name = "int"
    if min is not None and max is not None:
        handler_name += " between [{}, {}]".format(min, max)
    elif min is not None:
        handler_name += " >= {}".format(min)
    elif max is not None:
        handler_name += " <= {}".format(max)

    def type_handler(value):
        try:
            return _int(value)
        except:
            raise TypeError("Could not cast {} to an {}.".format(value, handler_name))

    type_handler.__name__ = handler_name
    return type_handler


_float = float


def float(min=None, max=None):
    """
        Type validator. Attempts to cast the value to an float, optionally within some
        bounds.

        :param min: Minimum valid value
        :type min: float
        :param max: Maximum valid value
        :type max: float
        :returns: Type validator function
        :raises: TypeError when value can't be cast.
        :rtype: function
    """
    handler_name = "float"
    if min is not None and max is not None:
        handler_name += " between [{}, {}]".format(min, max)
    elif min is not None:
        handler_name += " >= {}".format(min)
    elif max is not None:
        handler_name += " <= {}".format(max)

    def type_handler(value):
        try:
            return _int(value)
        except:
            raise TypeError("Could not cast {} to an {}.".format(value, handler_name))

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
        :returns: Type validator function
        :rtype: callable
    """

    if expand is None:
        expand = lambda x: [1.0] * x

    def type_handler(value):
        # No try block: let it raise the cast error.
        v = scalar_type(value)
        # Expand the scalar.
        return expand(v)

    type_handler.__name__ = "expanded list of " + scalar_type.__name__
    return type_handler


_list = list


def list(type=str, size=None):
    """
        Type validator for lists. Type casts each element to the given type and optionally
        validates the length of the list.

        :param type: Type validator of the elements.
        :type type: function
        :param size: Mandatory length of the list.
        :type size: int
        :returns: Type validator function
        :rtype: function
    """

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
    """
        Type validator. Type casts the value into a rational number between 0 and 1
        (inclusive).

        :returns: Type validator function
        :rtype: function
    """

    def type_handler(value):
        v = _float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError("{} is out of the 0-1 range for a fraction.".format(value))
        return v

    type_handler.__name__ = "fraction [0.; 1.]"
    return type_handler


def deg_to_radian():
    """
        Type validator. Type casts the value from degrees to radians.

        :returns: Type validator function
        :rtype: function
    """

    def type_handler(value):
        v = _float(value)
        return _float(v) * 2 * math.pi / 360

    type_handler.__name__ = "degrees"
    return type_handler
