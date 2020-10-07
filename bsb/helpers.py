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


def listify_input(value):
    """
    Turn any non-list values into a list containing the value. Sequences will be
    converted to a list using `list()`, `None` will  be replaced by an empty list.
    """
    # Replace None by empty array
    value = value if value is not None else []
    # Is `value` not a list?
    if not isinstance(value, (collections.abc.Sequence, np.ndarray)) or isinstance(
        value, str
    ):
        # Encapsulate any non-list input to a 1 element list
        value = [value]
    else:
        # Turn the sequence into a Python list.
        value = list(value)
    # Return listified value
    return value


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
            return n
        else:
            return list(map(map_1d_array, a))

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
