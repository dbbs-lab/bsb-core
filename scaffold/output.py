from .helpers import ConfigurableClass, get_qualified_class_name
from contextlib import contextmanager
from abc import abstractmethod
import h5py, time, numpy as np
from .morphologies import Morphology

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

class MorphologyRepository(OutputFormatter):

    defaults = {
        'file': 'morphology_repository.hdf5'
    }

    def __init__(self, file=None):
        super().__init__()
        if not file is None:
            self.file = file

    def get_handle(self):
        return h5py.File(self.file)

    def release_handle(self, handle):
        return handle.close()

    def save(self):
        pass

    def import_swc(self, file, name, tags=[]):
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
        with self.load() as f:
            dset = f.create_dataset(name, data=dataset_data)
            dset.attrs['name'] = name

    def get_morphology(self, name):
        '''
            Load a morphology from repository data
        '''
        # Open repository and close afterwards
        with self.load() as repo:
            # Check if morphology exists
            if not name in repo:
                raise Exception("Attempting to load unknown morphology '{}'".format(name))
            # Take out all the data with () index, and send along the metadata stored in the attributes
            return Morphology.from_repo_data(repo[name][()], repo[name].attrs)


    def init_scaffold(self):
        pass

    def validate(self):
        pass

class HDF5Formatter(OutputFormatter):

    defaults = {
        'file': 'scaffold_network_{}.hdf5'.format(time.strftime("%Y_%m_%d-%H%M%S"))
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

    def store_cell_connections(self, cells_group):
        connections_group = cells_group.create_group('connections')
        for tag, connectome_data in self.scaffold.cell_connections_by_tag.items():
            related_types = list(filter(lambda x: tag in x.tags, self.scaffold.configuration.connection_types.values()))
            connection_dataset = connections_group.create_dataset(tag, data=connectome_data)
            connection_dataset.attrs['tag'] = tag
            connection_dataset.attrs['connection_types'] = list(map(lambda x: x.name, related_types))
            connection_dataset.attrs['connection_type_classes'] = list(map(get_qualified_class_name, related_types))

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
