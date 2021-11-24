import abc, numpy as np, os, sys, collections
from contextlib import contextmanager
from inspect import isclass
import inspect, site
from .exceptions import *


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_config_path(file=None):
    packaged_configs = os.path.join(os.path.dirname(__file__), "configurations")
    global_install_configs = os.path.join(sys.prefix, "configurations")
    user_install_configs = os.path.join(site.USER_BASE, "configurations")
    if os.path.exists(packaged_configs):
        configs = packaged_configs
    elif os.path.exists(global_install_configs):
        configs = global_install_configs
    elif os.path.exists(user_install_configs):
        configs = user_install_configs
    else:
        raise FileNotFoundError("Could not locate configuration directory.")
    if file is not None:
        return os.path.join(configs, file)
    else:
        return configs


def get_qualified_class_name(x):
    return x.__class__.__module__ + "." + str(x.__class__.__name__)


class ConfigurableClass(abc.ABC):
    """
    A class that can be configured.
    """

    def initialise(self, scaffold):
        self.scaffold = scaffold
        self.cast_config()
        self.boot()
        self.validate()

    def boot(self):
        pass

    @abc.abstractmethod
    def validate(self):
        """
        Must be implemented by child classes. Raise exceptions when invalid configuration parameters
        are received.
        """
        pass

    def fill(self, conf, excluded=[]):
        self._raw_config = conf
        for name, prop in conf.items():
            if name not in excluded:
                self.__dict__[name] = prop

    def resolve_required(self, raw_required):
        requirements = []
        for required_condition in raw_required:
            if callable(required_condition):
                resolved = required_condition(self.__dict__)
                if resolved is None:
                    continue
                requirements.extend(listify_input(resolved))
            else:
                requirements.append(required_condition)
        return requirements

    def cast_config(self):
        """
        Casts/validates values imported onto this object from configuration files to their final form.
        The `casts` dictionary should contain the key of the attribute and a function that takes
        a value as only argument. This dictionary will be used to cast the attributes when cast_config
        is called.
        """
        name = ""
        if hasattr(self, "node_name"):
            name += self.node_name + "."
        if hasattr(self, "name"):
            name += self.name
        else:
            name = str(self)
        castingDict = getattr(self.__class__, "casts", {})
        defaultDict = getattr(self.__class__, "defaults", {})
        required = self.resolve_required(getattr(self.__class__, "required", []))

        # Get unique keys
        attrKeys = set([*castingDict.keys(), *defaultDict.keys(), *required])
        for attr in attrKeys:
            isRequired = attr in required
            hasDefault = attr in defaultDict
            shouldCast = attr in castingDict
            if not hasattr(self, attr):
                if hasDefault:
                    default_value = defaultDict[attr]
                    if isinstance(default_value, dict):
                        self.__dict__[attr] = default_value.copy()
                    elif isinstance(default_value, list):
                        self.__dict__[attr] = list(default_value)
                    else:
                        self.__dict__[attr] = default_value
                elif isRequired:
                    raise AttributeMissingError(
                        "Required attribute '{}' missing from '{}' section.".format(
                            attr, name
                        )
                    )
            elif shouldCast:
                cast = castingDict[attr]
                self.__dict__[attr] = cast_node(self.__dict__[attr], cast, attr, name)


def cast_node(value, cast, attr, name):
    if type(cast) is tuple:
        for union_cast in cast:
            # Try casting to each type in the union. Follows order.
            try:
                return cast_node(value, union_cast, attr, name)
            except Exception as e:
                pass
        # If this code path is reached, it means none of the casts succeeded without
        # an error so we should raise an error that the union cast failed.
        raise_union_cast(value, cast, attr, name)
    elif type(cast) is list:
        if len(cast) != 1:
            raise CastConfigurationError(
                "Invalid list casting configuration of {} in {}: can only cast a one-element list. The one element being the casting type of the list elements.".format(
                    attr, name
                )
            )
        cast = cast[0]
        # Try casting value to a list
        value = try_cast(value, list, attr, name)
        # Try casting each element of value to the cast type
        for i in range(len(value)):
            value[i] = cast_node(value[i], cast, attr + "[{}]".format(i), name)
        return value
    elif type(cast) is dict:
        raise NotImplementedError("Dictionary casting not implemented yet.")
    else:
        return try_cast(value, cast, attr, name)


def try_cast(value, cast, attr, name):
    try:
        # Try to cast using the specified cast function.
        v = cast(value)
        return v
    except Exception as e:
        if isinstance(
            e, ConfigurableCastError
        ):  # Is this an error raised by a child configurable class?
            # Format context and pass along the child cast exception.
            raise e.__class__("{}.{}: ".format(name, attr) + str(e)) from None
        # Use the function name, unless it is a class method called 'cast', then use the class name
        cast_name = (
            cast.__name__
            if not hasattr(cast, "__self__") or cast.__name__ != "cast"
            else cast.__self__.__name__
        )
        # Else, replace by generic "we couldn't" error.
        raise CastError(
            "{}.{}: Could not cast '{}' to a {}".format(name, attr, value, cast_name)
        )


def raise_union_cast(value, cast, attr, name):
    cast_names = []
    for c in cast:
        if c.__name__ == "cast" and hasattr(c, "__self__"):
            cast_names.append(c.__self__.__name__)
        else:
            cast_names.append(c.__name__)
    raise UnionCastError(
        "{}.{}: Could not cast '{}' to any of the following: {}".format(
            name, attr, value, ", ".join(cast_names)
        )
    )


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
    except:
        return [value]


class CastableConfigurableClass(ConfigurableClass):

    excluded = []

    @classmethod
    def cast(cast_class, value):
        class_instance = cast_class()
        class_instance.fill(value, cast_class.excluded)
        class_instance.cast_config()
        class_instance.boot()
        class_instance.validate()
        return class_instance


class OptionallyCastable(CastableConfigurableClass):
    @classmethod
    def cast(cast_class, value):
        class_instance = cast_class()
        if isinstance(value, cast_class):
            return value
        elif isinstance(value, dict):  # Configured by dictionary
            class_instance.type = "class"
            class_instance.fill(value, cast_class.excluded)
            class_instance.cast_config()
        else:  # Try fallback constant casting
            if not hasattr(cast_class, "fallback"):
                raise ConfigurableCastError(
                    "OptionallyCastable configuration classes require a fallback cast function. Missing for '{}'".format(
                        cast_class.__name__
                    )
                )
            try:
                value = cast_class.fallback(value)
            except Exception as e:
                raise
            class_instance.type = "const"
            class_instance.value = value
        class_instance.boot()
        class_instance.validate()
        return class_instance


class DistributionConfiguration(OptionallyCastable):
    """
    Cast a configuration node into a `scipy.stats` distribution.
    """

    # Fall back to float casting if no dictionary is given.
    fallback = float
    casts = {"type": str}
    required = ["type"]

    def validate(self):
        from scipy.stats import distributions

        if self.type == "const":
            return
        if self.type[-4:] == "_gen":
            raise InvalidDistributionError(
                "Distributions can not be created through their constructors but need to use their factory methods. (Those do not end in _gen)"
            )
        if self.type not in dir(distributions):
            raise UnknownDistributionError(
                "'{}' is not a distribution of scipy.stats".format(self.type)
            )
        try:
            distribution_factory = distributions.__dict__[self.type]
            distribution_kwargs = self._raw_config.copy()
            del distribution_kwargs["type"]
            self.distribution = distribution_factory(**distribution_kwargs)
        except TypeError as e:
            error_msg = str(e).replace(
                "_parse_args()", "scipy.stats.distributions." + self.type
            )
            raise InvalidDistributionError(error_msg) from None

    def draw(self, n):
        if self.type == "const":
            return [self.value for _ in range(n)]
        else:
            return self.distribution.rvs(size=n)

    def sample(self):
        return self.draw(1)[0]

    def mean(self):
        return self.distribution.mean()


class EvalConfiguration(OptionallyCastable):

    casts = {"statement": str, "variables": dict}
    required = ["statement"]

    def eval(self, locals=None):
        if self.type == "const":
            return self.value
        else:
            locals = {} if locals is None else locals
            globals = {"np": np}
            if hasattr(self, "variables"):
                locals.update(self.variables)
            result = eval(self.statement, globals, locals)
            return result

    def validate(self):
        pass


class ListEvalConfiguration(EvalConfiguration):
    fallback = list


class FloatEvalConfiguration(EvalConfiguration):
    fallback = float


def assert_attr(section, attr, section_name):
    if attr not in section:
        raise AttributeMissingError(
            "Required attribute '{}' missing in '{}'".format(attr, section_name)
        )
    return section[attr]


def if_attr(section, attr, default_value):
    if attr not in section:
        return default_value
    return section[attr]


def assert_strictly_one(section, attrs, section_name):
    attr_list = []
    for attr in attrs:
        if attr in section:
            attr_list.append(attr)
    if len(attr_list) != 1:
        msg = "{} found: ".format(len(attr_list)) + ", ".join(attr_list)
        if len(attr_list) == 0:
            msg = "None found."
        raise AttributeError(
            "Strictly one of the following attributes is expected in {}: {}. {}".format(
                section_name, ", ".join(attrs), msg
            )
        )
    else:
        return attr_list[0], section[attr_list[0]]


def assert_float(val, section_name):
    try:
        ret = float(val)
    except ValueError as e:
        raise TypeError("Invalid float '{}' given for '{}'".format(val, section_name))
    return ret


def assert_array(val, section_name):
    from collections.abc import Sequence

    if isinstance(val, Sequence):
        return val
    raise TypeError("Invalid array '{}' given for '{}'".format(val, section_name))


def assert_attr_float(section, attr, section_name):
    if attr not in section:
        raise AttributeMissingError(
            "Required attribute '{}' missing in '{}'".format(attr, section_name)
        )
    return assert_float(section[attr], "{}.{}".format(section_name, attr))


def assert_attr_array(section, attr, section_name):
    """
    Asserts that an attribute exists on a dictionary or object, and that it is an
    array.

    :param section: Dictionary or object that needs to contain the attribute.
    :type section: dict, object
    :param attr: Attribute name.
    :type attr: string
    :param section_name: Name of the section to print out the location of the missing attribute.
    :type section_name: string
    """
    if isinstance(section, dict) and attr in section:
        return assert_array(section[attr], "{}.{}".format(section_name, attr))
    elif hasattr(section, attr):
        return assert_array(section.__dict__[attr], "{}.{}".format(section_name, attr))

    raise AttributeMissingError(
        "Required attribute '{}' missing in '{}'".format(attr, section_name)
    )


def assert_attr_in(section, attr, values, section_name):
    """
    Assert that the attribute is present in the section dictionary and that its value is included
    in the given array.
    """
    if attr not in section:
        raise AttributeMissingError(
            "Required attribute '{}' missing in '{}'".format(attr, section_name)
        )
    if not section[attr] in values:
        raise AttributeError(
            "Attribute '{}.{}' with value '{}' must be one of the following values: {}".format(
                section_name, attr, section[attr], "'" + "', '".join(values) + "'"
            )
        )
    return section[attr]


class dimensions:
    def __init__(self, dimensions=None):
        self.dimensions = np.array([0.0, 0.0, 0.0]) if dimensions is None else dimensions

    @property
    def width(self):
        return self.dimensions[0]

    @property
    def height(self):
        return self.dimensions[1]

    @property
    def depth(self):
        return self.dimensions[2]

    @property
    def volume(self):
        return np.prod(self.dimensions)


class origin:
    def __init__(self, origin=None):
        self.origin = np.array([0.0, 0.0, 0.0]) if origin is None else origin

    def X(self):
        return self.origin[0]

    @property
    def Y(self):
        return self.origin[1]

    @property
    def Z(self):
        return self.origin[2]


class SortableByAfter:
    @abc.abstractmethod
    def has_after(self):
        pass

    @abc.abstractmethod
    def create_after(self):
        pass

    @abc.abstractmethod
    def get_after(self):
        pass

    @abc.abstractmethod
    def get_ordered(self, objects):
        pass

    def add_after(self, after_item):
        if not self.has_after():
            self.create_after()
        self.get_after().append(after_item)

    def is_after_satisfied(self, objects):
        """
        Determine whether the `after` specification of this cell type is met.
        Any cell types appearing in `self.after` need to occur before this cell type,
        so that this cell type appears "after" all these cell types.
        """
        if not self.has_after():  # No after?
            # Condition without constraints always True.
            return True
        is_met = False
        after = self.get_after()
        # Determine whether this cell type is out of order.
        for type in objects:
            if is_met and type.name in after:
                # After conditions not met if we have we seen ourself and
                # find something that's supposed to be in front of us.
                return False
            elif type == self:  # Is this us?
                # From this point on, nothing that appears in the after array is allowed to be encountered
                is_met = True
        # We didn't meet anything behind us that was supposed to be in front of us
        # => Condition met.
        return True

    def satisfy_after(self, objects):
        """
        Given an array of cell types, place this cell type after all of the
        cell types specified in `self.after`. If cell types in `self.after`
        are missing from the given array this cell type is placed at the end
        of the array. Modifies the `objects` array in place.
        """
        before_types = self.get_after().copy()
        i = 0
        place_after = False
        while len(before_types) > 0 and i < len(objects):
            if objects[i].name in before_types:
                before_types.remove(objects[i].name)
            if objects[i] == self:
                objects.remove(self)
                place_after = True
            else:
                i += 1
        if place_after:
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
        after_specifications = list(filter(lambda c: c.has_after(), sorting_objects))
        j = 0
        # Keep rearranging as long as any cell type's after condition isn't satisfied.
        while any(
            map(lambda c: not c.is_after_satisfied(sorting_objects), after_specifications)
        ):
            j += 1
            # Rearrange each element that is out of place.
            for after_type in after_specifications:
                if not after_type.is_after_satisfied(sorting_objects):
                    after_type.satisfy_after(sorting_objects)
            # If we have had to rearrange all elements more than there are elements, the
            # conditions cannot be met, and a circular dependency is at play.
            if j > len(objects):
                raise OrderError(
                    "Couldn't resolve order, probably a circular dependency including: {}".format(
                        ", ".join(
                            list(
                                map(
                                    lambda c: c.name,
                                    filter(
                                        lambda c: not c.is_after_satisfied(
                                            sorting_objects
                                        ),
                                        after_specifications,
                                    ),
                                )
                            )
                        )
                    )
                )
        # Return the sorted array.
        return sorting_objects


def map_ndarray(data, _map=None):
    if _map is None:
        _map = []
    last_index = -1
    last_value = None

    def map_1d_array(e):
        nonlocal last_index, last_value, _map
        if last_index == -1 or e != last_value:
            try:
                last_index = _map.index(e)
            except ValueError as ex:
                last_index = len(_map)
                _map.append(e)
            last_value = e
        return last_index

    def n_dim_map(a):
        n = np.empty(a.shape, dtype=int)
        if len(a.shape) > 1:
            for i, b in enumerate(a):
                n[i] = n_dim_map(b)
        else:
            n[:] = list(map(map_1d_array, a))
        return n

    _mapped = n_dim_map(data)
    return _mapped, _map


def load_configurable_class(name, configured_class_name, parent_class, parameters={}):
    if isclass(configured_class_name):
        instance = configured_class_name(**parameters)
    else:
        class_ref = get_configurable_class(configured_class_name)
        if not parent_class is None and not issubclass(class_ref, parent_class):
            raise DynamicClassError(
                "Configurable class '{}' must derive from {}.{}".format(
                    configured_class_name,
                    parent_class.__module__,
                    parent_class.__qualname__,
                )
            )
        instance = class_ref(**parameters)
    instance.__dict__["name"] = name
    return instance


def fill_configurable_class(obj, conf, excluded=[]):
    for name, prop in conf.items():
        if not name in excluded:
            if isinstance(prop, dict) or isinstance(prop, list):
                prop = prop.copy()
            obj.__dict__[name] = prop


def get_configurable_class(configured_class_name):
    class_parts = configured_class_name.split(".")
    class_name = class_parts[-1]
    module_name = ".".join(class_parts[:-1])
    if module_name == "":
        module_dict = globals()
    else:
        import os, sys

        sys.path.insert(0, os.getcwd())
        module_ref = __import__(module_name, globals(), locals(), [class_name], 0)
        sys.path = sys.path[1:]
        module_dict = module_ref.__dict__
    if not class_name in module_dict:
        raise ConfigurableClassNotFoundError("Class not found: " + configured_class_name)
    return module_dict[class_name]


def continuity_list(iterable, step=1):
    """
    Return a compacted notation of a list of nearly continuous numbers.

    The ``iterable`` will be iterated and chains of continuous numbers will be
    determined. Each chain will then be added to the output format as a starting
    number and count.

    *Example:* ``[4,5,6,7,8,9,12]`` ==> ``[4,6,12,1]``

    :param iterable: The collection of elements to be compacted.
    :type iterable: iter
    :param step: ``iterable[i]`` needs to be equal to ``iterable[i - 1] + step`` for
      them to considered continuous.
    """
    serial = []
    iterator = iter(iterable)
    # Get the first item in the iterator to set the initial conditions.
    try:
        item = next(iterator)
    # Iterable is empty.
    except StopIteration:
        # Return empty serialization.
        return []
    # Initial condition.
    start = item
    last_item = item
    counter = 1
    # Iterate over the remaining items.
    for item in iterator:
        # Is this new item discontinuous with the last?
        if item != last_item + step:
            # End the chain at the previous item and start a new one.
            # Store the start and length of the last chain in the serialized list.
            serial.append(start)
            serial.append(counter)
            # Set the values for a new list
            counter = 1
            start = item
        else:
            counter += 1
        last_item = item
    # Finish the last chain
    serial.append(start)
    serial.append(counter)
    return serial


def continuity_hop(iterator):
    """
    Hop over a continuity list in steps of 2, returning the start & count pairs.
    """
    try:
        while True:
            yield (next(iterator), next(iterator))
    except StopIteration:
        pass


def expand_continuity_list(iterable, step=1):
    """
    Return the full set of items associated with the continuity list, as formatted by
    :func:`.helpers.continuity_list`.
    """
    deserialized = []
    for start, count in continuity_hop(iter(iterable)):
        # Each hop, expand a `count` long list starting from `start`
        # taking `step` sized steps, and append it to the deserialized list.
        end = start + count * step
        deserialized.extend(list(range(start, end, step)))
    return deserialized


def iterate_continuity_list(iterable, step=1):
    """
    Generate the continuity list
    """
    for start, count in continuity_hop(iter(iterable)):
        end = start + count * step
        for i in range(start, end, step):
            yield i


def count_continuity_list(iterable):
    total = 0
    for _, count in continuity_hop(iter(iterable)):
        # Each hop, add the count to the total
        total += count
    return total
