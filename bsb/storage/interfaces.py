import abc
import types
import functools
from contextlib import contextmanager


class Interface(abc.ABC):
    def __init__(self, handler):
        self._handler = handler


class Engine(Interface):
    def __init__(self, root):
        self.root = root

    @property
    def format(self):
        # This attribute is set on the engine by the storage provider and correlates to
        # the name of the engine plugin.
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


class PlacementSet(Interface):
    @abc.abstractmethod
    def __init__(self, engine, type):
        pass

    @abc.abstractclassmethod
    def create(self, engine, type):
        """
        Override with a method to create the placement set.
        """
        pass

    @abc.abstractstaticmethod
    def exists(self, engine, type):
        """
        Override with a method to check existence of the placement set
        """
        pass

    def require(self, engine, type):
        """
        Can be overridden with a method to make sure the placement set exists. The default
        implementation uses the class's ``exists`` and ``create`` methods.
        """
        if not self.exists(engine, type):
            self.create(engine, type)

    @abc.abstractmethod
    def get_all_chunks(self):
        pass

    @abc.abstractproperty
    def load_positions(self):
        """
        Return a dataset of cell positions.
        """
        pass

    @abc.abstractproperty
    def load_rotations(self):
        """
        Return a dataset of cell rotations.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        pass

    @abc.abstractproperty
    def load_morphologies(self):
        """
        Return a :class:`~.storage.interfaces.MorphologySet` associated to the cells.

        :return: Set of morphologies
        :rtype: :class:`~.storage.interfaces.MorphologySet`

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def append_data(self, chunk, positions=None, morphologies=None):
        pass

    @abc.abstractmethod
    def append_additional(self, name, chunk, data):
        pass


class MorphologyRepository(Interface):
    @abc.abstractmethod
    def select(self, selector):
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


class StoredMorphology:
    def __init__(self, name, loader, meta):
        self.name = name
        self._loader = loader
        self._meta = meta

    def get_meta(self):
        return self._meta.copy()

    def load(self):
        return self._loader()

    @functools.cache
    def cached_load(self):
        return self.load()
