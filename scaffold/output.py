from .helpers import ConfigurableClass
from contextlib import contextmanager
from abc import abstractmethod
import h5py, time

class OutputFormatter(ConfigurableClass):
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

class HDF5Formatter(OutputFormatter):

    defaults = {
        'file': 'scaffold_network_{}.hdf5'.format(time.strftime("%Y_%m_%d-%H%M%S"))
    }

    def get_handle(self):
        return h5py.File(self.file, 'r')

    def release_handle(self, handle):
        return handle.close()

    def save(self):
        self.storage = h5py.File(self.file, 'w')
        self.store_configuration()
        self.store_cells()
        self.store_statistics()
        self.store_appendices()
        self.storage.close()

    def validate(self):
        pass

    def store_configuration(self):
        f = self.storage
        f.attrs['configuration_name'] = self.scaffold.configuration._name
        f.attrs['configuration_type'] = self.scaffold.configuration._type
        f.attrs['configuration_string'] = self.scaffold.configuration._raw

    def store_cells(self):
        cells_group = self.storage.create_group('cells')
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
            connection_dataset.attrs['connection_type_classes'] = list(map(lambda x: str(x.__class__), related_types))

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
