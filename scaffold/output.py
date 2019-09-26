from .helpers import ConfigurableClass, get_qualified_class_name
from contextlib import contextmanager
from abc import abstractmethod
import h5py, os, time, pickle, numpy as np
from .morphologies import Morphology
from numpy import string_

class OutputFormatter(ConfigurableClass):

    def __init__(self):
        super().__init__()
        self.save_file_as = None
        self.storage = None

    @contextmanager
    def load(self):
        handle = self.get_handle()
        try:
            yield handle
        finally:
            self.release_handle(handle)

    @abstractmethod
    def get_handle(self):
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
    def load_tree(self, collection_name, tree_name):
        '''
            Load a tree from a tree collection in the storage
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

class MorphologyRepository(OutputFormatter):

    defaults = {
        'file': 'morphology_repository.hdf5'
    }

    def __init__(self, file=None):
        super().__init__()
        self.handle = None
        if not file is None:
            self.file = file

    def get_handle(self):
        '''
            Open the MorphologyRepository storage resource.
        '''
        if not self.handle is None: # Resource already open?
            # Return the handle to the already open resource.
            return self.handle
        # Open a new handle to the resource.
        self.handle = h5py.File(self.file)
        # Repository structure missing from resource? Create it.
        if not 'morphologies' in self.handle:
            self.handle.create_group('morphologies')
        if not 'morphologies/voxel_clouds' in self.handle:
            self.handle.create_group('morphologies/voxel_clouds')
        # Return the handle to the resource.
        return self.handle

    def release_handle(self, handle):
        '''
            Close the MorphologyRepository storage resource.
        '''
        self.handle = None
        return handle.close()

    def save(self):
        '''
            Called when the scaffold is saving itself.
            Don't need to do anything special with the repo when the scaffold is saving itself.
        '''
        pass

    def import_swc(self, file, name, tags=[], overwrite=False):
        '''
            Import and store .swc file contents as a morphology in the repository.
        '''
        # Read as CSV
        swc_data = np.loadtxt(file)
        # Create empty dataset
        dataset_length = len(swc_data)
        dataset_data = np.empty((dataset_length, 10))
        # Map parent id's to start coordinates. Root node (id: -1) is at 0., 0., 0.
        starts = {-1: [0., 0., 0.]}
        # Iterate over the compartments
        for i in range(dataset_length):
            # Extract compartment record
            compartment = swc_data[i, :]
            compartment_id = compartment[0]
            compartment_type = compartment[1]
            compartment_parent = compartment[6]
            # Check if parent id is known
            if not compartment_parent in starts:
                raise Exception("Node {} references a parent node {} that isn't know yet".format(compartment_id, compartment_parent))
            # Use parent endpoint as startpoint, get endpoint and store it as a startpoint for child compartments
            compartment_start = starts[compartment_parent]
            compartment_end = compartment[2:5]
            starts[compartment_id] = compartment_end
            # Get more compartment radius
            compartment_radius = compartment[5]
            # Store compartment in the repository dataset
            dataset_data[i] = [
                compartment_id,
                compartment_type,
                *compartment_start,
                *compartment_end,
                compartment_radius,
                compartment_parent
            ]
        # Save the dataset in the repository
        with self.load() as repo:
            if overwrite: # Do we overwrite previously existing dataset with same name?
                self._rmm(name) # Delete anything that might be under this name.
            elif self._me(name):
                raise Exception("A morphology called '{}' already exists in this repository.")
            # Create the dataset
            dset = repo['morphologies'].create_dataset(name, data=dataset_data)
            # Set attributes
            dset.attrs['name'] = name
            dset.attrs['type'] = 'swc'

    def get_morphology(self, name):
        '''
            Load a morphology from repository data
        '''
        # Open repository and close afterwards
        with self.load() as repo:
            # Check if morphology exists
            if not self._me(name):
                raise Exception("Attempting to load unknown morphology '{}'".format(name))
            # Take out all the data with () index, and send along the metadata stored in the attributes
            data = self._m(name)
            return Morphology.from_repo_data(data[()], data.attrs)

    def morphology_exists(self, name):
        with self.load() as repo:
            return self._me(name)

    def voxel_cloud_exists(self, name):
        with self.load() as repo:
            return self._ve(name)

    def remove_morphology(self, name):
        with self.load() as repo:
            self._rmm(name)

    def remove_voxel_cloud(self, name):
        with self.load() as repo:
            self._rmv(name)

    def list_all_morphologies(self):
        with self.load() as repo:
            return list(filter(lambda x: x != 'voxel_clouds', repo['morphologies'].keys()))

    def list_all_voxelized(self):
        with self.load() as repo:
            all = list(repo['morphologies'].keys())
            voxelized = list(filter(lambda x: x in repo['/morphologies/voxel_clouds'], all))
            return voxelized

    ## Handle avoidance factories
    ##   These function are shorthands for internal use that assume an open handle and don't close the handle.

    def _me(self, name):
        '''
            Shorthand for self.morphology_exists
        '''
        return name in self.handle['morphologies']

    def _ve(self, name):
        '''
            Shorthand for self.voxel_cloud_exists
        '''
        return name in self.handle['morphologies/voxel_clouds']

    def _rmm(self, name):
        '''
            Shorthand for self.remove_morphology
        '''
        if self._me(name):
            del self.handle['morphologies/' + name]

    def _rmv(self, name):
        '''
            Shorthand for self.remove_voxel_cloud
        '''
        if self._ve(name):
            del self.handle['morphologies/voxel_clouds' + name]

    def _m(self, name):
        '''
            Return the morphology dataset
        '''
        return self.handle['morphologies/' + name]

    ## Parts of the interface that we don't need. Should restructure to a pure OutputHandler for the handles
    ## and an extension OutputFormatter for integration with the scaffold

    def init_scaffold(self):
        pass

    def validate(self):
        pass

    def load_tree(self, collection_name, tree_name):
        pass

    def get_simulator_output_path(self, simulator_name):
        pass

    def has_cells_of_type(self, name):
        pass

    def get_cells_of_type(self, name):
        pass

class HDF5Formatter(OutputFormatter):

    defaults = {
        'file': 'scaffold_network_{}.hdf5'.format(time.strftime("%Y_%m_%d-%H%M%S")),
        'simulator_output_path': False
    }

    def get_handle(self):
        return h5py.File(self.file, 'r+')

    def release_handle(self, handle):
        return handle.close()

    def save(self):
        if self.save_file_as:
            self.storage = h5py.File(self.save_file_as, 'w')
            self.file = self.save_file_as
        else:
            self.storage = h5py.File(self.file, 'w')
        self.store_configuration()
        self.store_cells()
        self.store_trees()
        self.store_statistics()
        self.store_appendices()
        self.storage.close()

    def init_scaffold(self):
        with self.load() as resource:
            self.scaffold.configuration.cell_type_map = resource['cells'].attrs['types']
            self.scaffold.placement_stitching = resource['cells/stitching'][:]
            for cell_type_name, count in resource['statistics/cells_placed'].attrs.items():
                self.scaffold.statistics.cells_placed[cell_type_name] = count

    def validate(self):
        pass

    def store_configuration(self):
        f = self.storage
        f.attrs['shdf_version'] = 3.0
        f.attrs['configuration_version'] = 3.0
        f.attrs['configuration_name'] = self.scaffold.configuration._name
        f.attrs['configuration_type'] = self.scaffold.configuration._type
        f.attrs['configuration_class'] = get_qualified_class_name(self.scaffold.configuration)
        f.attrs['configuration_string'] = self.scaffold.configuration._raw

    def store_cells(self):
        cells_group = self.storage.create_group('cells')
        cells_group.create_dataset('stitching', data=self.scaffold.placement_stitching)
        self.store_cell_positions(cells_group)
        self.store_cell_connections(cells_group)

    def store_cell_positions(self, cells_group):
        position_dataset = cells_group.create_dataset('positions', data=self.scaffold.cells)
        cell_type_names = self.scaffold.configuration.cell_type_map
        cells_group.attrs['types'] = cell_type_names
        for type in self.scaffold.configuration.cell_types.keys():
            position_dataset.attrs[type + '_map'] = np.where(self.scaffold.cells[:,1] == cell_type_names.index(type))[0]

    def store_cell_connections(self, cells_group):
        connections_group = cells_group.create_group('connections')
        for tag, connectome_data in self.scaffold.cell_connections_by_tag.items():
            related_types = list(filter(lambda x: tag in x.tags, self.scaffold.configuration.connection_types.values()))
            connection_dataset = connections_group.create_dataset(tag, data=connectome_data)
            connection_dataset.attrs['tag'] = tag
            connection_dataset.attrs['connection_types'] = list(map(lambda x: x.name, related_types))
            connection_dataset.attrs['connection_type_classes'] = list(map(get_qualified_class_name, related_types))

    def store_trees(self):
        tree_group = self.storage.create_group('trees')
        for tree_collection_name, tree_collection in self.scaffold.trees.__dict__.items():
            tree_collection_group = tree_group.create_group(tree_collection_name)
            for tree_name, tree in tree_collection.items():
                tree_dataset = tree_collection_group.create_dataset(tree_name, data=string_(pickle.dumps(tree)))

    def store_statistics(self):
        statistics = self.storage.create_group('statistics')
        self.store_placement_statistics(statistics)

    def store_placement_statistics(self, statistics_group):
        storage_group = statistics_group.create_group('cells_placed')
        for key, value in self.scaffold.statistics.cells_placed.items():
            storage_group.attrs[key] = value

    def store_appendices(self):
        # Append extra datasets specified internally or by user.
        for key, data in self.scaffold.appends.items():
            dset = self.storage.create_dataset(key, data=data)

    def load_tree(self, collection_name, tree_name):
        with self.load() as f:
            try:
                return pickle.loads(f['/trees/{}/{}'.format(collection_name, tree_name)][()])
            except KeyError as e:
                raise Exception("Tree not found in HDF5 file '{}', path does not exist: '{}'".format(f.file))

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
            return resource['/cells/positions'][resource['/cells/positions'].attrs[name + '_map']][()]
