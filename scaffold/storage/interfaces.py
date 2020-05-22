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


class Configuration(Interface):
    @abc.abstractmethod
    def store(self, configuration):
        pass

    @abc.abstractmethod
    def load(self):
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
            new_class._filters = {ft: [] for ft in new_class.get_filter_types()}
        return new_class


class Filter(Interface, metaclass=_FilterMeta):
    def __init__(self, handler, filters):
        super().__init__(handler)
        self.filters = filters

    @classmethod
    @abc.abstractmethod
    def get_filter_types(cls):
        pass

    @classmethod
    def create(cls, handler, **kwargs):
        f = cls(handler, kwargs)
        return f

    def activate(self):
        for f in self.filters:
            try:
                if self not in self.__class__._filters[f]:
                    self.__class__._filters[f].append(self)
            except KeyError:
                raise NotImplemented(
                    "The '{}' engine does not support the '{}' filter.".format(
                        self._handler._format, f
                    )
                )

    def deactivate(self):
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
        try:
            return [f.filters[filter_type] for f in cls._filters[filter_type]]
        except KeyError:
            pass
