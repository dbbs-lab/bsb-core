import abc
import builtins
import inspect
import math
from weakref import WeakKeyDictionary

import numpy as np

from ..exceptions import (
    CastError,
    ClassMapMissingError,
    InvalidReferenceError,
    NoneReferenceError,
    RequirementError,
    TypeHandlingError,
)
from ._compile import _reserved_kw_passes, _wrap_reserved
from ._make import _load_object


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


def any_():
    def type_handler(value):
        return value

    type_handler.__name__ = "any"
    return type_handler


def in_(container):
    """
    Type validator. Checks whether the given value occurs in the given container.
    Uses the `in` operator.

    :param container: List of possible values
    :type container: list
    :returns: Type validator function
    :rtype: Callable
    """
    error_msg = "a value in: " + builtins.str(container)

    def type_handler(value):
        if value in container:
            return value
        else:
            raise TypeError(f"Couldn't cast '{value}' to " + error_msg)

    type_handler.__name__ = error_msg
    return type_handler


def or_(*type_args):
    """
    Type validator. Attempts to cast the value to any of the given types in order.

    :param type_args: Another type validator
    :type type_args: Callable
    :returns: Type validator function
    :raises: TypeError if none of the given type validators can cast the value.
    :rtype: Callable
    """
    handler_name = "any of: " + ", ".join(x.__name__ for x in type_args)
    # Make sure to wrap all type handlers so that they accept the parent and key args.
    type_args = [_wrap_reserved(t) for t in type_args]

    def type_handler(value, _parent=None, _key=None):
        type_errors = {}
        for t in type_args:
            try:
                v = t(value, _parent=_parent, _key=_key)
            except Exception as e:
                type_error = (
                    builtins.str(e.__class__.__module__)
                    + "."
                    + builtins.str(e.__class__.__name__)
                    + ": "
                    + builtins.str(e)
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


class object_(TypeHandler):
    """
    Type validator. Attempts to import the value, absolute, or relative to the
    `module_path` entries.

    :param module_path: List of the modules that should be searched when doing a
      relative import.
    :type module_path: list[str]
    :raises: TypeError when value can't be cast.
    :returns: Type validator function
    :rtype: Callable
    """

    def __init__(self, module_path=None):
        self._module_path = module_path

    def __call__(self, value):
        msg = f"Could not import '{value}': "
        try:
            if isinstance(value, builtins.str):
                obj = _load_object(value, self._module_path)
                obj._cfg_inv = value
            else:
                obj = value
        except Exception as e:
            raise TypeError(msg + builtins.str(e))
        return obj

    def __inv__(self, value):
        return getattr(value, "_cfg_inv", value)

    @property
    def __name__(self):
        return "object"


class class_(object_):
    """
    Type validator. Attempts to import the value as the name of a class, relative to
    the `module_path` entries, absolute or just returning it if it is already a class.

    :param module_path: List of the modules that should be searched when doing a
      relative import.
    :type module_path: list[str]
    :raises: TypeError when value can't be cast.
    :returns: Type validator function
    :rtype: Callable
    """

    def __call__(self, value):
        if inspect.isclass(value):
            return value
        obj = super().__call__(value)
        if not inspect.isclass(obj):
            raise TypeError(f"'{value}' is not a class, got {builtins.type(obj)} instead")
        return obj

    def __inv__(self, value):
        if not inspect.isclass(value):
            value = type(value)
        return f"{value.__module__}.{value.__name__}"

    @property
    def __name__(self):
        return "class"


class function_(object_):
    """
    Type validator. Attempts to import the value, absolute, or relative to the
    `module_path` entries, and verifies that it is callable.

    :param module_path: List of the modules that should be searched when doing a
      relative import.
    :type module_path: list[str]
    :raises: TypeError when value can't be cast.
    :returns: Type validator function
    :rtype: Callable
    """

    def __call__(self, value):
        if callable(value):
            return value
        obj = super().__call__(value)
        if not callable(obj):
            raise TypeError(f"Could not import {value} as a callable function.")
        return obj

    def __inv__(self, value):
        return f"{value.__module__}.{value.__name__}"

    @property
    def __name__(self):
        return "function"


class method(function_):
    def __init__(self, class_name):
        super().__init__()
        self._class = class_name

    def __call__(self, value):
        parent = class_()(self._class)
        try:
            obj = getattr(parent, value)
        except AttributeError as e:
            raise TypeError(builtins.str(e)) from None
        if not callable(obj):
            raise TypeError(f"Could not import '{value}' as a method of `{self._class}`.")
        return obj

    def __inv__(self, value):
        return value.__name__

    @property
    def __name__(self):
        return f"method of '{self._class}'"


class WeakInverter:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._map = WeakKeyDictionary()

    def store_value(self, value, result):
        self._map[result] = value

    def __inv__(self, value):
        return self._map.get(value, value)


class method_shortcut(method, function_):
    def __call__(self, value):
        try:
            obj = method.__call__(self, value)
        except TypeError:
            try:
                obj = function_.__call__(self, value)
            except TypeError:
                raise TypeError(
                    f"Could not import '{value}' as a function or a method of `{self._class}`."
                ) from None
        return obj

    def __inv__(self, value):
        if inspect.isfunction(value):
            try:
                method.__call__(self, value.__name__)
                return method.__inv__(self, value)
            except TypeError:
                return function_.__inv__(self, value)
        else:
            return value


def str(strip=False, lower=False, upper=False, safe=True):
    """
    Type validator. Attempts to cast the value to a str, optionally with some sanitation.

    :param strip: Trim whitespaces
    :type strip: bool
    :param lower: Convert value to lowercase
    :type lower: bool
    :param upper: Convert value to uppercase
    :type upper: bool
    :param safe: If False, checks that the type of value is string before cast.
    :type safe: bool
    :returns: Type validator function
    :raises: TypeError when value can't be cast.
    :rtype: Callable
    """
    handler_name = "str"
    # Compile a custom function to sanitize the string according to args
    safety_check = "\n if not isinstance(s, str):\n  raise TypeError()\n" if safe else ""
    fstr = f"def f(s):{safety_check} return str(s)"
    for add, mod in zip((strip, lower, upper), ("strip", "lower", "upper")):
        if add:
            fstr += f".{mod}()"
    go_fish = builtins.dict()
    exec(compile(fstr, "__", "exec"), go_fish)
    f = go_fish["f"]

    def type_handler(value):
        return f(value)

    type_handler.__name__ = handler_name
    return type_handler


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
    :rtype: Callable
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
            v = builtins.int(value)
            if min is not None and min > v or max is not None and max < v:
                raise Exception()
            return v
        except Exception:
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
    :rtype: Callable
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
        except Exception:
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
    :rtype: Callable
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
            if isinstance(value, builtins.int):
                v = builtins.int(value)
                if min is not None and min > v or max is not None and max < v:
                    raise Exception()
            else:
                v = _float(value)
                if min is not None and min > v or max is not None and max < v:
                    raise Exception()
            return v
        except Exception:
            raise TypeError("Could not cast {} to a {}.".format(value, handler_name))

    type_handler.__name__ = handler_name
    return type_handler


def key():
    """
    Type handler for keys in configuration trees. Keys can be either int indices of a
    config list, or string keys of a config dict.

    :returns: Type validator function
    :rtype: Callable
    """

    def type_handler(value):
        if not (isinstance(value, builtins.int) or isinstance(value, builtins.str)):
            raise TypeError(f"{type(value)} is not an int or str")
        else:
            return value

    type_handler.__name__ = "configuration key"
    return type_handler


def scalar_expand(scalar_type, size=None, expand=None):
    """
    Create a method that expands a scalar into an array with a specific size or uses
    an expansion function.

    :param scalar_type: Type of the scalar
    :type scalar_type: type
    :param size: Expand the scalar to an array of a fixed size.
    :type size: int
    :param expand: A function that takes the scalar value as argument and returns the
      expanded form.
    :type expand: Callable
    :returns: Type validator function
    :rtype: Callable
    """

    if expand is None:

        def expand(x):
            return [1.0] * x

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
    :rtype: Callable
    """
    type_handler = or_(list(scalar_type, size), scalar_type)

    type_handler.__name__ += " or " + scalar_type.__name__
    return type_handler


def voxel_size():
    return list_or_scalar(float(), 3)


def list(type=builtins.str, size=None):
    """
    Type validator for lists. Type casts each element to the given type and optionally
    validates the length of the list.

    :param type: Type validator of the elements.
    :type type: Callable
    :param size: Mandatory length of the list.
    :type size: int
    :returns: Type validator function
    :rtype: Callable
    """

    def type_handler(value):
        # Simple lists default to returning None for None, while configuration lists
        # default to an empty list.
        if value is None:
            return None
        v = builtins.list(value)
        try:
            for i, e in enumerate(v):
                v[i] = type(e)
        except Exception:
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


def dict(type=builtins.str):
    """
    Type validator for dicts. Type casts each element to the given type.

    :param type: Type validator of the elements.
    :type type: Callable
    :returns: Type validator function
    :rtype: Callable
    """

    def type_handler(value):
        if value is None:
            return None
        v = builtins.dict(value)
        try:
            for k, e in v.items():
                v[k] = type(e)
        except Exception:
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
    :rtype: Callable
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


class distribution(TypeHandler):
    """
    Type validator. Type casts the value or node to a distribution.
    """

    def __call__(self, value, _key=None, _parent=None):
        from ._distributions import Distribution

        if not isinstance(value, builtins.list) and not isinstance(value, builtins.dict):
            value = {"distribution": "constant", "constant": value}

        return Distribution(**value, _key=_key, _parent=_parent)

    @property
    def __name__(self):  # pragma: nocover
        return "distribution"

    def __inv__(self, value):
        if value["distribution"] == "constant":
            return value["constant"]
        else:
            return value


class evaluation(TypeHandler):
    """
    Type validator. Provides a structured way to evaluate a python statement from the
    config. The evaluation context provides ``numpy`` as ``np``.

    :returns: Type validator function
    :rtype: Callable
    """

    def __init__(self):
        self._references = {}

    def __call__(self, value):
        cfg = builtins.dict(value)
        statement = builtins.str(cfg.get("statement", "None"))
        locals_ = builtins.dict(cfg.get("variables", {}))
        globals_ = {"np": np}
        res = eval(statement, globals_, locals_)
        self._references[id(res)] = value
        return res

    @property
    def __name__(self):
        return "evaluation"

    def get_original(self, value):
        """
        Return the original configuration node associated with the given evaluated value.

        :param value: A value that was produced by this type handler.
        :type value: Any
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
    :rtype: Callable
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
            raise CastError(
                f"'{value}' is not a valid classmap identifier for `{class_name}`."
                + f" Choose from: {classmap_str}"
            )
        return value

    type_handler.__name__ = "a classmap value"
    return type_handler


def mut_excl(*mutuals, required=True, max=1, shortform=False):
    """
    Requirement handler for mutually exclusive attributes.

    :param str mutuals: The keys of the mutually exclusive attributes.
    :param bool required: Whether at least one of the keys is required
    :param int max: The maximum amount of keys that may occur together.
    :param bool shortform: Allow the short form alternative.
    :returns: Requirement function
    :rtype: Callable
    """
    listed = ", ".join(f"`{m}`" for m in mutuals[:-1])
    if len(mutuals) > 1:
        listed += f" {{}} `{mutuals[-1]}`"

    def requirement(section):
        if shortform and section.is_shortform:
            return False
        bools = [m in section for m in mutuals]
        given = sum(bools)
        if given > max:
            if max > 1:
                err_msg = f"Maximum {max} of {listed} may be specified. {given} given."
            else:
                err_msg = f"The {listed} attributes are mutually exclusive."
            err_msg = err_msg.format("and")
            raise RequirementError(err_msg)
        if not given and required:
            err_msg = f"A {listed} attribute is required."
            raise RequirementError(err_msg)
        return False

    return requirement


def same_size(*list_attrs, required=True):
    """
    Requirement handler for list attributes that should have the same size.

    :param list_attrs: The keys of the list attributes.
    :type list_attrs: str
    :param required: Whether at least one of the keys is required
    :type required: bool
    :returns: Requirement function
    :rtype: Callable
    """
    listed = ", ".join(f"`{m}`" for m in list_attrs[:-1])
    if len(list_attrs) > 1:
        listed += f" {{}} `{list_attrs[-1]}`"

    def requirement(section):
        common_size = -1
        count = 0
        for m in list_attrs:
            if m in section:
                v = builtins.list(section[m])
                if len(v) != common_size and common_size >= 0:
                    err_msg = f"The {listed} attributes should have the same size."
                    raise RequirementError(err_msg)
                common_size = len(v)
                count += 1
        if not count == len(list_attrs) and required:
            err_msg = f"The {listed} attributes are required."
            raise RequirementError(err_msg)
        return False

    return requirement


def shortform():
    def requirement(section):
        return not section.is_shortform

    return requirement


class ndarray(TypeHandler):
    """
    Type validator numpy arrays.

    :returns: Type validator function
    :rtype: Callable
    """

    def __init__(self, shape: builtins.tuple[builtins.int, ...] = None, dtype=None):
        """
        :param shape: shape of the array, optional.
        :param dtype: data-type, optional
        """
        if any(dim < 0 for dim in (shape or ())):
            raise TypeError(f"Ndarray shape must all be positive. Provided {shape}.")
        self.shape = shape
        self.dtype = dtype

    def __call__(self, value):
        result = np.array(value, copy=False)
        if self.dtype is not None:
            result = np.asarray(result, dtype=self.dtype)
        if self.shape is not None:
            try:
                result = result.reshape(self.shape)
            except Exception:
                raise TypeError(
                    f"Couldn't cast array of {getattr(value, 'shape', 'unknown')} shape into an array of {self.shape} shape."
                )
        return result

    @property
    def __name__(self):
        return "ndarray"

    def __inv__(self, value):
        return value.tolist()


def none():
    def type_handler(value, _parent, _key=None):
        if value is not None:
            raise TypeError("value is not None")
        return value

    type_handler.__name__ = "a None value"
    return type_handler


class PackageRequirement(TypeHandler):
    def __call__(self, value):
        from packaging.requirements import Requirement

        requirement = Requirement(value)
        requirement._cfg_inv = value
        return requirement

    @property
    def __name__(self):
        return "package requirement"

    def __inv__(self, value):
        return getattr(value, "_cfg_inv", builtins.str(value))

    def __hint__(self):
        return "numpy==1.24.0"


__all__ = [
    "PackageRequirement",
    "TypeHandler",
    "WeakInverter",
    "any_",
    "class_",
    "deg_to_radian",
    "dict",
    "distribution",
    "evaluation",
    "float",
    "fraction",
    "function_",
    "in_",
    "in_classmap",
    "int",
    "key",
    "list",
    "list_or_scalar",
    "method",
    "method_shortcut",
    "mut_excl",
    "ndarray",
    "none",
    "number",
    "object_",
    "or_",
    "scalar_expand",
    "shortform",
    "str",
    "voxel_size",
]
__api__ = ["PackageRequirement", "TypeHandler", "WeakInverter"]
