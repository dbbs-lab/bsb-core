from ..exceptions import *
from ._hooks import overrides
from ._make import _load_class
import math, sys, numpy as np, abc, functools, weakref
from inspect import signature as _inspect_signature

_any = any
_reserved_keywords = ["_parent", "_key"]
_list = list


class TypeHandler(abc.ABC):
    """
    Base class for any type handler that cannot be described as a single function.

    Declare the `__call__(self, value)` method to convert the given value to the
    desired type, raising a `TypeError` if it failed in an expected manner.

    Declare the `__name__(self)` method to return a name for the type handler to
    display in messages to the user such as errors.

    Declare the optional `__inv__` method to invert the given value back to its
    original value, the type of the original value will usually be lost but the type
    of the returned value can still serve as a suggestion.
    """

    @abc.abstractmethod
    def __call__(self, value):  # pragma: nocover
        pass

    @property
    @abc.abstractmethod
    def __name__(self):  # pragma: nocover
        return "unknown type handler"

    def __inv__(self, value):  # pragma: nocover
        return value

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        call = cls.__call__
        passes = _reserved_kw_passes(call)
        if not all(passes.values()):
            cls.__call__ = _wrap_reserved(call)


def _reserved_kw_passes(f):
    # Inspect the signature and wrap the typecast in a wrapper that will accept and
    # strip the missing 'key' kwarg
    try:
        sig = _inspect_signature(f)
        params = sig.parameters
    except:
        params = []

    return {key: key in params for key in _reserved_keywords}


def _wrap_reserved(t):
    """
    Wrap a type handler in a wrapper that accepts all reserved keyword arguments that
    the config system will push into the type handler call, and pass only those that
    the original type handler accepts. This way type handlers can accept any
    combination of the reserved keyword args without raising TypeErrors when they do
    not accept one.
    """
    # Type handlers never need to be wrapped. The `__init_subclass__` of the TypeHandler
    # class handles wrapping of `__call__` implementations so that they accept and strip
    # _parent & _key.
    if isinstance(t, TypeHandler):
        return t

    # Check which reserved keywords the function already takes
    passes = _reserved_kw_passes(t)
    if all(passes.values()):
        return t

    # Create the keyword arguments of the outer function that accepts all reserved kwargs
    reserved_keys = "".join(f", {key}=None" for key in _reserved_keywords)
    header = f"def type_handler(value, *args{reserved_keys}, **kwargs):\n"
    passes = "".join(f", {key}={key}" for key in _reserved_keywords if passes[key])
    # Create the call to the inner function that is passed only the kwargs that it accepts
    wrap = f" return orig(value, *args{passes}, **kwargs)"
    # Compile the code block and indicate that the function was compiled here.
    mod = compile(header + wrap, f"{__file__}/<_wrap_reserved:compile>", "exec")
    # Execute the code block in this local scope and pick the function out of the scope
    exec(mod, {"orig": t}, bait := locals())
    type_handler = bait["type_handler"]
    # Copy over the metadata of the original function
    type_handler = functools.wraps(t)(type_handler)
    type_handler.__name__ = t.__name__
    return type_handler


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
    # Make sure to wrap all type handlers so that they accept the parent and key args.
    type_args = [_wrap_reserved(t) for t in type_args]

    def type_handler(value, _parent=None, _key=None):
        type_errors = {}
        for t in type_args:
            try:
                v = t(value, _parent=_parent, _key=_key)
            except Exception as e:
                type_error = (
                    str(e.__class__.__module__)
                    + "."
                    + str(e.__class__.__name__)
                    + ": "
                    + str(e)
                )
                type_errors[t.__name__] = type_error
            else:

                return v
        type_errors = "\n".join(
            "- Casting to '{}' raised:\n{}".format(n, e) for n, e in type_errors.items()
        )
        # Use a CastError instead of a TypeError so that the message is passed along as is
        # by upstream error handlers.
        raise CastError(
            "Couldn't cast {} into {}.\n{}".format(value, handler_name, type_errors)
        )

    type_handler.__name__ = handler_name
    return type_handler


def class_(module_path=None):
    """
    Type validator. Attempts to import the value as the name of a class, relative to
    the `module_path` entries, absolute or just returning it if it is already a class.

    :param module_path: List of the modules that should be searched when doing a
      relative import.
    :type module_path: list of module
    :returns: Type validator function
    :raises: TypeError when value can't be cast.
    :rtype: function
    """

    def type_handler(value):
        try:
            return _load_class(value, module_path)
        except:
            raise TypeError("Could not import {} as a class".format(value))

    type_handler.__name__ = "class"
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
            v = _int(value)
            if min is not None and min > v or max is not None and max < v:
                raise Exception()
            return v
        except:
            raise TypeError(
                "Could not cast {} to an {}.".format(value, handler_name)
            ) from None

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
            v = _float(value)
            if min is not None and min > v or max is not None and max < v:
                raise Exception()
            return v
        except:
            raise TypeError("Could not cast {} to an {}.".format(value, handler_name))

    type_handler.__name__ = handler_name
    return type_handler


def number(min=None, max=None):
    """
    Type validator. If the given value is an int returns an int, tries to cast to float
    otherwise

    :param min: Minimum valid value
    :type min: float
    :param max: Maximum valid value
    :type max: float
    :returns: Type validator function
    :raises: TypeError when value can't be cast.
    :rtype: function
    """
    handler_name = "number"
    if min is not None and max is not None:
        handler_name += " between [{}, {}]".format(min, max)
    elif min is not None:
        handler_name += " >= {}".format(min)
    elif max is not None:
        handler_name += " <= {}".format(max)

    def type_handler(value):
        try:
            if isinstance(value, _int):
                v = _int(value)
                if min is not None and min > v or max is not None and max < v:
                    raise Exception()
            else:
                v = _float(value)
                if min is not None and min > v or max is not None and max < v:
                    raise Exception()
            return v
        except:
            raise TypeError("Could not cast {} to a {}.".format(value, handler_name))

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


def list_or_scalar(scalar_type, size=None):
    """
    Type validator that accepts a scalar or list of said scalars.

    :param scalar_type: Type of the scalar
    :type scalar_type: type
    :param size: Expand the scalar to an array of a fixed size.
    :type size: int
    :returns: Type validator function
    :rtype: callable
    """
    type_handler = or_(list(scalar_type, size), scalar_type)

    type_handler.__name__ += " or " + scalar_type.__name__
    return type_handler


def voxel_size():
    return list_or_scalar(float(), 3)


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
            return None
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


_dict = dict


def dict(type=str):
    """
    Type validator for dicts. Type casts each element to the given type.

    :param type: Type validator of the elements.
    :type type: function
    :returns: Type validator function
    :rtype: function
    """

    def type_handler(value):
        if value is None:
            return None
        v = _dict(value)
        try:
            for k, e in v.items():
                v[k] = type(e)
        except:
            raise TypeError(
                "Couldn't cast {} of {} into {}".format(k, value, type.__name__)
            )
        return v

    type_handler.__name__ = "dict of {}".format(type.__name__)
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


class deg_to_radian(TypeHandler):
    """
    Type validator. Type casts the value from degrees to radians.
    """

    def __call__(self, value):
        v = _float(value)
        return v * 2 * math.pi / 360

    @property
    def __name__(self):  # pragma: nocover
        return "degrees"

    def __inv__(self, value):
        v = _float(value)
        return v * 360 / (2 * math.pi)


class _ConstantDistribution:
    def __init__(self, const):
        self.const = const

    def draw(self, n):
        return np.ones(n) * self.const

    def __tree__(self):
        return self.const


def constant_distr():
    """
    Type handler that turns a float into a distribution that always returns the float.
    This can be used in places where a distribution is expected but the user might
    want to use a single constant value instead.

    :returns: Type validator function
    :rtype: function
    """

    def type_handler(value):
        return _ConstantDistribution(_float(value))

    type_handler.__name__ = "constant distribution"
    return type_handler


def distribution():
    """
    Type validator. Type casts a float to a constant distribution or a _dict to a
    :class:`Distribution <.config.nodes.Distribution>` node.

    :returns: Type validator function
    :rtype: function
    """
    from .nodes import Distribution

    return or_(constant_distr(), Distribution)


class evaluation(TypeHandler):
    """
    Type validator. Provides a structured way to evaluate a python statement from the
    config. The evaluation context provides ``numpy`` as ``np``.

    :returns: Type validator function
    :rtype: function
    """

    def __init__(self):
        self._references = {}

    def __call__(self, value):
        cfg = _dict(value)
        statement = str(cfg.get("statement", "None"))
        locals = _dict(cfg.get("variables", {}))
        globals = {"np": np}
        res = eval(statement, globals, locals)
        self._references[id(res)] = value
        return res

    @property
    def __name__(self):
        return "evaluation"

    def get_original(self, value):
        """
        Return the original configuration node associated with the given evaluated value.

        :param value: A value that was produced by this type handler.
        :type value: any
        :raises: NoneReferenceError when `value` is `None`, InvalidReferenceError when
          there is no config associated to the object id of this value.
        """
        # None is a singleton, so it's not bijective, it's also the value returned when
        # a weak reference is removed; so it's doubly unsafe to check for references to it
        if value is None:
            raise NoneReferenceError("Can't create bijection for NoneType value.")
        vid = id(value)
        # Create a set of references from our stored weak references that are still alive.
        if vid not in self._references:
            raise InvalidReferenceError(f"No evaluation reference found for {vid}", value)
        return self._references[vid]

    def __inv__(self, value):
        try:
            return self.get_original(value)
        except TypeHandlingError:
            # Original does not exist or can't be obtained, just return the given value.
            return value


def in_classmap():
    """
    Type validator. Checks whether the given string occurs in the class map of a
    dynamic node.

    :returns: Type validator function
    :rtype: function
    """

    def type_handler(value, _parent, _key=None):
        class_name = _parent.__class__.__name__
        if not hasattr(_parent.__class__, "_config_dynamic_classmap"):
            raise ClassMapMissingError(
                f"Class map missing for `{class_name}`,"
                + " required when using `in_classmap` type handler."
            )
        classmap = _parent.__class__._config_dynamic_classmap
        if value not in classmap:
            classmap_str = ", ".join(f"'{key}'" for key in classmap)
            raise TypeError(
                f"'{value}' is not a valid classmap identifier for `{class_name}`."
                + f" Choose from: {classmap_str}"
            )
        return value

    type_handler.__name__ = "a classmap value"
    return type_handler
