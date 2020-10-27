import abc, types
from contextlib import contextmanager


class Interface(abc.ABC):
    def __init__(self, handler):
        self._handler = handler


class Engine(Interface):
    def __init__(self, resource_identifier):
        self.resource_identifier = resource_identifier

    @property
    def format(self):
        # This attribute is set on the engine by the storage provider and correlates to
        # the name of the engine directory.
        return self._format

    @abc.abstractmethod
    def exists(self):
        pass

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def move(self, new_root):
        pass

    @abc.abstractmethod
    def remove(self):
        pass


class NetworkDescription(Interface):
    pass


class ConfigStore(Interface):
    @abc.abstractmethod
    def store(self, cso):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def get_parser_name(self):
        pass


class TreeCollectionHandler(Interface):
    """
    Interface that allows a Engine to handle storage of TreeCollections.
    """

    @abc.abstractmethod
    def load_tree(collection_name, tree_name):
        pass

    @abc.abstractmethod
    def store_tree_collections(self, tree_collections):
        pass

    @abc.abstractmethod
    def list_trees(self, collection_name):
        pass


class PlacementSet(Interface):
    @abc.abstractmethod
    def __init__(self, handler, type):
        pass

    @abc.abstractclassmethod
    def create(self, handler, type):
        pass

    @abc.abstractstaticmethod
    def exists(self, handler, type):
        pass

    @abc.abstractproperty
    def identifiers(self):
        """
        Return a list of cell identifiers.
        """
        pass

    @abc.abstractproperty
    def positions(self):
        """
        Return a dataset of cell positions.
        """
        pass

    @abc.abstractproperty
    def rotations(self):
        """
        Return a dataset of cell rotations.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        pass

    @abc.abstractproperty
    def cells(self):
        """
        Reorganize the available datasets into a collection of :class:`Cells
        <.models.Cell>`
        """
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def append_data(self, identifiers, positions=None, rotations=None):
        pass

    @abc.abstractmethod
    def append_cells(self, cells):
        pass


class ConnectivitySet(Interface):
    pass


class Label(Interface):
    @abc.abstractmethod
    def label(self, identifiers):
        pass

    @abc.abstractmethod
    def unlabel(self, identifiers):
        pass

    @abc.abstractmethod
    def store(self, identifiers):
        pass

    @property
    @abc.abstractmethod
    def cells(self):
        pass

    @abc.abstractmethod
    def list(self):
        pass


class _FilterMeta(abc.ABCMeta, type):
    """
    Metaclass for the Filter interface: Uses the abstract `get_filter_types` method to
    set the `_filters` class attribute.
    """

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        # If the direct base class is the Interface class then we're creating the parent
        # class below, and the metaclass changes should not be applied.
        if bases[0].__name__ != "Interface":
            new_class._filters = {}
        return new_class


class Filter(Interface, metaclass=_FilterMeta):
    """
    Filters are object that store multiple key value pairs, the keys being filter types and the
    values the filter to apply to that filter type. Filters can be activated and deactivated.

    Other pieces of code can then query the Filter class
    (``Filter.get_filters(type)``) to get all active filters of a type and use it to
    filter their operation.

    Filter objects can be used as context managers.
    """

    def __init__(self, handler, filters):
        super().__init__(handler)
        self.__class__._init_handler(handler)
        self.filters = filters

    @classmethod
    def _init_handler(cls, handler):
        if handler not in cls._filters:
            cls._filters[handler] = {ft: {} for ft in cls.get_filter_types()}

    @classmethod
    @abc.abstractmethod
    def get_filter_types(cls):
        """
        This method should be overridden with a class method to return all the
        available filter types on the engine as a list of strings.
        """
        pass

    @classmethod
    def create(cls, handler, **kwargs):
        """
        Create a multifilter object.
        """
        f = cls(handler, kwargs)
        return f

    def activate(self):
        """
        Activate a multifilter object
        """
        filters = self.__class__._filters[self._handler]
        for f in self.filters:
            try:
                if self not in filters[f]:
                    filters[f].append(self)
            except KeyError:
                raise NotImplemented(
                    "The '{}' engine does not support the '{}' filter.".format(
                        self._handler._format, f
                    )
                )

    def deactivate(self):
        """
        Deactivate a multifilter object.
        """
        for f in self.filters:
            try:
                self.__class__._filters[f].remove(self)
            except KeyError:
                pass

    def __enter__(self):
        self.activate()

    def __exit__(self):
        self.deactivate()

    @classmethod
    def get_filters(cls, filter_type):
        """
        Return all active filters of the given type.
        """
        if filter_type not in cls._filters[self._handler]:
            raise NotImplemented(
                "The '{}' engine does not support the '{}' filter.".format(
                    self._handler._format, f
                )
            )
        filters = []
        for f in cls._filters[filter_type]:
            if filter_type in f.filters:
                filters.append(f)
        return filters
