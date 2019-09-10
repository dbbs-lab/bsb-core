from .helpers import ConfigurableClass
from abc import abstractmethod

class OutputFormatter(ConfigurableClass):
    @abstractmethod
    def save(self):
        pass

class HDF5Formatter(OutputFormatter):

    def save(self):
        self.storage = h5py.File('scaffold_new_test.hdf5', 'w')
        self.store_configuration()
        self.store_cells()
        self.store_statistics()
        self.store_appendices()
        f.close()

    def validate(self):
        pass

    def store_configuration(self):
        f = self.storage
        f.attrs['configuration_type'] = self.scaffold.configuration._type
        f.attrs['configuration_string'] = self.scaffold.configuration._raw

    def store_cells(self):
        cells_group = self.storage.create_group('cells')
        self.store_cell_positions(cells_group)
        self.store_cell_connections(cells_group)
        # # Store the entire connectivity matrix
        # self.storage.create_dataset('connectome', data=self.scaffold.cell_connections)

    def store_cell_positions(self, cells_group):
        position_dataset = cells_group.create_dataset('positions', data=self.scaffold.cells)
        cell_type_names = self.scaffold.configuration.cell_type_map
        cells_group.attrs['types'] = cell_type_names

    def store_cell_connections(self, cells_group):
        connections_group = cells_group.create_group('connections')
        for connection_type_name, connectome_data in self.scaffold.cell_connections_by_type.items():
            connection_dataset = connections_group.create_dataset(connection_type_name, data=connectome_data)
            connection_dataset.attrs['name'] = connection_type_name

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
