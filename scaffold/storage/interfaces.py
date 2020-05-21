import abc
from contextlib import contextmanager


class Interface(abc.ABC):
    def __init__(self, handler):
        self._handler = handler


class Engine(Interface):
    def __init__(self, resource_identifier):
        self.resource_identifier = resource_identifier

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
    def add(self, identifiers):
        pass

    @abc.abstractmethod
    def add(self, identifiers):
        pass


class Filter(Interface):
    @abc.abstractmethod
    def activate(self):
        pass

    @abc.abstractmethod
    def deactivate(self):
        pass
