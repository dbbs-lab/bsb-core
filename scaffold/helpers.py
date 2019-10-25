import abc, inspect, numpy as np
from .exceptions import *

def get_qualified_class_name(x):
    return x.__class__.__module__ + '.' + str(x.__class__.__name__)

def copyIniKey(obj, section, key_config):
    ini_key = key_config['key']
    if not ini_key in section: # Only copy values that exist in the config
        return

    # Process the config values based on the type in their key_config.
    morph_map = {'micrometer': float, 'float': float, 'string': str}
    obj.__dict__[ini_key] = morph_map[key_config['type']](section[ini_key])


class ConfigurableClass(abc.ABC):
    '''
        A class that can be configured.
    '''

    def initialise(self, scaffold):
        self.scaffold = scaffold
        self.castConfig()
        self.boot()
        self.validate()

    def boot(self):
        pass

    @abc.abstractmethod
    def validate(self):
        '''
            Must be implemented by child classes. Raise exceptions when invalid configuration parameters
            are received.
        '''
        pass

    def fill(self, conf, excluded=[]):
        self._raw_config = conf
        print('added raw', self._raw_config)
        for name, prop in conf.items():
            if not name in excluded:
                self.__dict__[name] = prop

    def castConfig(self):
        '''
            Casts/validates values imported onto this object from configuration files to their final form.
            The `casts` dictionary should contain the key of the attribute and a function that takes
            a value as only argument. This dictionary will be used to cast the attributes when castConfig
            is called.
        '''
        name = ''
        if hasattr(self, 'node_name'):
            name += self.node_name + '.'
        if hasattr(self, 'name'):
            name += self.name
        else:
            name = str(self)
        castingDict = getattr(self.__class__, 'casts', {})
        defaultDict = getattr(self.__class__, 'defaults', {})
        required =    getattr(self.__class__, 'required', [])
        # Get unique keys
        attrKeys = set([*castingDict.keys(), *defaultDict.keys(), *required])
        for attr in attrKeys:
            isRequired = attr in required
            hasDefault = attr in defaultDict
            shouldCast = attr in castingDict
            if not hasattr(self, attr):
                if hasDefault:
                    self.__dict__[attr] = defaultDict[attr]
                elif isRequired:
                    raise Exception("Required attribute '{}' missing from '{}' section.".format(attr, name))
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
            raise Exception("Invalid list casting configuration of {} in {}: can only cast a one-element list. The one element being the casting type of the list elements.".format(attr, name))
        cast = cast[0]
        # Try casting value to a list
        value = try_cast(value, list, attr, name)
        # Try casting each element of value to the cast type
        for i in range(len(value)):
            value[i] = cast_node(value[i], cast, attr + '[{}]'.format(i), name)
        return value
    elif type(cast) is dict:
        raise Exception("Dictionary casting not implemented yet. (no use case)")
    else:
        return try_cast(value, cast, attr, name)

def try_cast(value, cast, attr, name):
    try:
        # Try to cast using the specified cast function.
        v = cast(value)
        return v
    except Exception as e:
        if isinstance(e, ConfigurableCastException): # Is this an error raised by a child configurable class?
            # Format context and pass along the child cast exception.
            raise e.__class__("{}.{}: ".format(name, attr) + str(e)) from None
        # Use the function name, unless it is a class method called 'cast', then use the class name
        cast_name = cast.__name__ if not hasattr(cast, "__self__") or cast.__name__ != "cast" else cast.__self__.__name__
        # Else, replace by generic "we couldn't" error.
        raise CastException("{}.{}: Could not cast '{}' to a {}".format(
            name, attr, value, cast_name
        ))

def raise_union_cast(value, cast, attr, name):
    cast_names = []
    for c in cast:
        if c.__name__ == "cast" and hasattr(c, '__self__'):
            cast_names.append(c.__self__.__name__)
        else:
            cast_names.append(c.__name__)
    raise UnionCastException("{}.{}: Could not cast '{}' to any of the following: {}".format(
        name, attr, value, ", ".join(cast_names)
    ))

class CastableConfigurableClass(ConfigurableClass):

    excluded = []

    @classmethod
    def cast(cast_class, value):
        class_instance = cast_class()
        class_instance.fill(value, cast_class.excluded)
        return class_instance

class DistributionConfiguration(CastableConfigurableClass):

    casts = {
        "mean": float,
        "sd": float,
        "type": str
    }

    @classmethod
    def cast(cast_class, value):
        class_instance = cast_class()
        if isinstance(value, dict): # Configured by dictionary
            class_instance.fill(value, cast_class.excluded)
            class_instance.validate()
        else: # Try for int configuration
            try:
                value = int(value)
            except Exception as e:
                raise
            class_instance.type = "const"
            class_instance.value = value
        return class_instance

    def validate(self):
        from scipy.stats import distributions
        if self.type[-4:] == "_gen":
            raise InvalidDistributionException("Distributions can not be created through their constructors but need to use their factory methods. (Those do not end in _gen)")
        if not self.type in dir(distributions):
            raise UnknownDistributionException("'{}' is not a distribution of scipy.stats".format(self.type))
        try:
            distribution_factory = distributions.__dict__[self.type]
            distribution_kwargs = self._raw_config.copy()
            del distribution_kwargs['type']
            self.distribution = distribution_factory(**distribution_kwargs)
        except TypeError as e:
            error_msg = str(e).replace("_parse_args()", "scipy.stats.distributions." + self.type)
            raise InvalidDistributionException(error_msg) from None

    def draw(self, n):
        if self.type == "const":
            return [self.value for _ in range(n)]
        else:
            return self.distribution.rvs(size=n)

    def sample(self):
        return self.draw(1)[0]

def assert_attr(section, attr, section_name):
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    return section[attr]

def if_attr(section, attr, default_value):
    if not attr in section:
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
        raise Exception("Strictly one of the following attributes is expected in {}: {}. {}".format(section_name, ", ".join(attrs), msg))
    else:
        return attr_list[0], section[attr_list[0]]

def assert_float(val, section_name):
    try:
        ret = float(val)
    except ValueError as e:
        raise Exception("Invalid float '{}' given for '{}'".format(val, section_name))
    return ret

def assert_array(val, section_name):
    from collections import Sequence
    if isinstance(val, Sequence):
        return val
    raise Exception("Invalid array '{}' given for '{}'".format(val, section_name))

def assert_attr_float(section, attr, section_name):
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    return assert_float(section[attr], "{}.{}".format(section_name, attr))

def assert_attr_array(section, attr, section_name):
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    return assert_array(section[attr], "{}.{}".format(section_name, attr))

def assert_attr_in(section, attr, values, section_name):
    '''
        Assert that the attribute is present in the section dictionary and that its value is included
        in the given array.
    '''
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    if not section[attr] in values:
        raise Exception("Attribute '{}.{}' with value '{}' must be one of the following values: {}".format(
            section_name,
            attr,
            section[attr],
            "'" + "', '".join(values) + "'"
        ))
    return section[attr]


class dimensions:
    def __init__(self, dimensions=None):
        self.dimensions = np.array([0., 0., 0.]) if dimensions is None else dimensions

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
        self.origin = np.array([0., 0., 0.]) if origin is None else origin

    def X(self):
        return self.origin[0]

    @property
    def Y(self):
        return self.origin[1]

    @property
    def Z(self):
        return self.origin[2]
