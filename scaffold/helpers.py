import abc, numpy as np

def get_qualified_class_name(x):
    return x.__class__.__module__ + '.' + str(x.__class__.__name__)

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
                    default_value = defaultDict[attr]
                    if isinstance(default_value, dict):
                        self.__dict__[attr] = default_value.copy()
                    elif isinstance(default_value, list):
                        self.__dict__[attr] = list(default_value)
                    else:
                        self.__dict__[attr] = default_value
                elif isRequired:
                    raise Exception("Required attribute '{}' missing from '{}' section.".format(attr, name))
            elif shouldCast:
                cast = castingDict[attr]
                def cast_node(value, cast, attr):
                    def try_cast(value, cast):
                        try:
                            return cast(value)
                        except Exception as e:
                            raise Exception("{}.{}: Could not cast '{}' to a {}".format(
                                name,
                                attr,
                                value,
                                cast.__name__
                            ))

                    if type(cast) is list:
                        if len(cast) != 1:
                            raise Exception("Invalid list casting configuration of {} in {}: can only cast a one-element list. The one element being the casting type of the list elements.".format(attr, name))
                        cast = cast[0]
                        # Try casting value to a list
                        value = try_cast(value, list)
                        # Try casting each element of value to the cast type
                        for i in range(len(value)):
                            value[i] = cast_node(value[i], cast, attr + '[{}]'.format(i))
                        return value
                    elif type(cast) is dict:
                        raise Exception("Dictionary casting not implemented yet. (no use case)")
                    else:
                        return try_cast(value, cast)

                self.__dict__[attr] = cast_node(self.__dict__[attr], cast, attr)

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

class SortableByAfter:
    @abc.abstractmethod
    def has_after(self):
        pass

    @abc.abstractmethod
    def get_after(self):
        pass

    @abc.abstractmethod
    def get_ordered(self, objects):
        pass

    def is_after_satisfied(self, objects):
        '''
            Determine whether the `after` specification of this cell type is met.
            Any cell types appearing in `self.after` need to occur before this cell type,
            so that this cell type appears "after" all these cell types.
        '''
        if not self.has_after(): # No after?
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
            elif type == self: # Is this us?
                # From this point on, nothing that appears in the after array is allowed to be encountered
                is_met = True
        # We didn't meet anything behind us that was supposed to be in front of us
        # => Condition met.
        return True

    def satisfy_after(self, objects):
        '''
            Given an array of cell types, place this cell type after all of the
            cell types specified in `self.after`. If cell types in `self.after`
            are missing from the given array this cell type is placed at the end
            of the array. Modifies the `objects` array in place.
        '''
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
        '''
            Orders a given dictionary of objects by the class's default mechanism and
            then apply the `after` attribute for further restrictions.
        '''
        # Sort by the default approach
        sorting_objects = cls.get_ordered(objects)
        # Afterwards cell types can be specified that need to be placed after other types.
        after_specifications = list(filter(lambda c: c.has_after(), objects.values()))
        j = 0
        # Keep rearranging as long as any cell type's after condition isn't satisfied.
        while any(map(lambda c: not c.is_after_satisfied(sorting_objects), after_specifications)):
            j += 1
            # Rearrange each element that is out of place.
            for after_type in after_specifications:
                if not after_type.is_after_satisfied(sorting_objects):
                    after_type.satisfy_after(sorting_objects)
            # If we have had to rearrange all elements more than there are elements, the
            # conditions cannot be met, and a circular dependency is at play.
            if j > len(objects):
                raise Exception("Couldn't resolve order, probably a circular dependency including: {}".format(
                    ", ".join(list(map(lambda c: c.name, filter(lambda c: not c.is_after_satisfied(sorting_objects), after_specifications))))))
        # Return the sorted array.
        return sorting_objects
