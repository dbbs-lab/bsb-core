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
                raise OrderError(
                    f"Couldn't resolve order, probably a circular dependency including: {circulars}"
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
