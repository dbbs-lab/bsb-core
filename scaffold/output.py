from .helpers import ConfigurableClass, get_qualified_class_name
from contextlib import contextmanager
from abc import abstractmethod, ABC
import h5py, os, time, pickle, numpy as np
from numpy import string_

class ResourceHandler(ABC):
    def __init__(self):
        self.handle = None

    @contextmanager
    def load(self, mode=None):
        already_open = False
        if self.handle is None: # Is the handle not open yet? Open it.
            # Pass the mode argument if it is given, otherwise allow child to rely
            # on its own default value for the mode argument.
            self.handle = self.get_handle(mode) if not mode is None else self.get_handle()
            already_open = True
        try:
            yield self.handle # Return the handle
        finally: # This is always called after the context manager closes.
            if not already_open: # Did we open the handle? We close it.
                self.release_handle(self.handle)
                self.handle = None

    @abstractmethod
    def get_handle(self, mode=None):
        '''
            Open the output resource and return a handle.
        '''
        pass

    @abstractmethod
    def release_handle(self, handle):
        '''
            Close the open output resource and release the handle.
        '''
        pass

class HDF5ResourceHandler(ResourceHandler):
    def get_handle(self, mode='r+'):
        '''
            Open an HDF5 resource.
        '''
        # Open a new handle to the resource.
        return h5py.File(self.file, mode)

    def release_handle(self, handle):
        '''
            Close the MorphologyRepository storage resource.
        '''
        return handle.close()

class TreeHandler(ResourceHandler):
    '''
        Interface that allows a ResourceHandler to handle storage of TreeCollections.
    '''

    @abstractmethod
    def load_tree(collection_name, tree_name):
        pass

    @abstractmethod
    def store_tree_collections(self, tree_collections):
        pass

class HDF5TreeHandler(HDF5ResourceHandler, TreeHandler):
    '''
        TreeHandler that uses HDF5 as resource storage
    '''
    def store_tree_collections(self, tree_collections):
        tree_group = self.handle.create_group('trees')
        for tree_collection in tree_collections:
            tree_collection_group = tree_group.create_group(tree_collection.name)
            for tree_name, tree in tree_collection.items():
                tree_dataset = tree_collection_group.create_dataset(tree_name, data=string_(pickle.dumps(tree)))

    def load_tree(self, collection_name, tree_name):
        with self.load() as f:
            try:
                return pickle.loads(f['/trees/{}/{}'.format(collection_name, tree_name)][()])
            except KeyError as e:
                raise Exception("Tree not found in HDF5 file '{}', path does not exist: '{}'".format(f.file))

class OutputFormatter(ConfigurableClass, TreeHandler):

    def __init__(self):
        ConfigurableClass.__init__(self)
        TreeHandler.__init__(self)
        self.save_file_as = None

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def init_scaffold(self):
        '''
            Initialize the scaffold when it has been loaded from an output file.
        '''
        pass

    @abstractmethod
    def get_simulator_output_path(self, simulator_name):
        '''
            Return the path where a simulator can dump preliminary output.
        '''
        pass

    @abstractmethod
    def has_cells_of_type(self, name):
        '''
            Check whether the position matrix for a certain cell type is present.
        '''
        pass

    @abstractmethod
    def get_cells_of_type(self, name):
        '''
            Return the position matrix for a specific cell type.
        '''
        pass

class HDF5Formatter(OutputFormatter, HDF5TreeHandler):

    defaults = {
        'file': 'scaffold_network_{}.hdf5'.format(time.strftime("%Y_%m_%d-%H%M%S")),
        'simulator_output_path': False
    }

    def save(self):
        if self.save_file_as:
            self.file = self.save_file_as

        with self.load('w') as output:
            self.store_configuration()
            self.store_cells()
            self.store_tree_collections(self.scaffold.trees.__dict__.values())
            self.store_statistics()
            self.store_appendices()

    def init_scaffold(self):
        with self.load() as resource:
            self.scaffold.configuration.cell_type_map = resource['cells'].attrs['types']
            self.scaffold.placement_stitching = resource['cells/stitching'][:]
            for cell_type_name, count in resource['statistics/cells_placed'].attrs.items():
                self.scaffold.statistics.cells_placed[cell_type_name] = count

    def validate(self):
        pass

    def store_configuration(self):
        f = self.handle
        f.attrs['shdf_version'] = 3.0
        f.attrs['configuration_version'] = 3.0
        f.attrs['configuration_name'] = self.scaffold.configuration._name
        f.attrs['configuration_type'] = self.scaffold.configuration._type
        f.attrs['configuration_class'] = get_qualified_class_name(self.scaffold.configuration)
        f.attrs['configuration_string'] = self.scaffold.configuration._raw

    def store_cells(self):
        cells_group = self.handle.create_group('cells')
        cells_group.create_dataset('stitching', data=self.scaffold.placement_stitching)
        self.store_cell_positions(cells_group)
        self.store_cell_connections(cells_group)

    def store_cell_positions(self, cells_group):
        position_dataset = cells_group.create_dataset('positions', data=self.scaffold.cells)
        cell_type_names = self.scaffold.configuration.cell_type_map
        cells_group.attrs['types'] = cell_type_names
        type_maps_group = cells_group.create_group('type_maps')
        for type in self.scaffold.configuration.cell_types.keys():
            type_maps_group.create_dataset(type + '_map', data=np.where(self.scaffold.cells[:,1] == cell_type_names.index(type))[0])

    def store_cell_connections(self, cells_group):
        connections_group = cells_group.create_group('connections')
        for tag, connectome_data in self.scaffold.cell_connections_by_tag.items():
            related_types = list(filter(lambda x: tag in x.tags, self.scaffold.configuration.connection_types.values()))
            connection_dataset = connections_group.create_dataset(tag, data=connectome_data)
            connection_dataset.attrs['tag'] = tag
            connection_dataset.attrs['connection_types'] = list(map(lambda x: x.name, related_types))
            connection_dataset.attrs['connection_type_classes'] = list(map(get_qualified_class_name, related_types))

    def store_statistics(self):
        statistics = self.handle.create_group('statistics')
        self.store_placement_statistics(statistics)

    def store_placement_statistics(self, statistics_group):
        storage_group = statistics_group.create_group('cells_placed')
        for key, value in self.scaffold.statistics.cells_placed.items():
            storage_group.attrs[key] = value

    def store_appendices(self):
        # Append extra datasets specified internally or by user.
        for key, data in self.scaffold.appends.items():
            dset = self.handle.create_dataset(key, data=data)

    def get_simulator_output_path(self, simulator_name):
        return self.simulator_output_path or os.getcwd()

    def has_cells_of_type(self, name):
        with self.load() as resource:
            return name in list(resource['/cells'].attrs['types'])

    def get_cells_of_type(self, name):
        # Check if cell type is present
        if not self.has_cells_of_type(name):
            raise Exception("Attempting to load cell type '{}' that isn't defined in the storage.".format(name))
        # Slice out the cells of this type based on the map in the position dataset attributes.
        with self.load() as resource:
            type_map = self.get_type_map(name)
            return resource['/cells/positions'][type_map][()]

    def get_type_map(self, type):
        with self.load() as resource:
            return self.handle['/cells/type_maps/{}_map'.format(type)][()]
