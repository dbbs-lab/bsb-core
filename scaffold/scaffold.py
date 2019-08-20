from .statistics import Statistics
from .plotting import plotNetwork
import numpy as np
import h5py
from pprint import pprint
import time

###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads geometries, morphologies, ...
#    * Creates network architecture
#    * Sets up simulation

class Scaffold:

	def __init__(self, config):
		self.configuration = config
		self.resetNetworkCache()
		# Debug statistics, unused.
		self.statistics = Statistics(self)
		self._nextId = 0
		# Use the configuration to initialise all components such as cells and layers
		# to prepare for the network architecture compilation.
		self.initialiseComponents()
		# Code to be compliant with old code, to be removed after rework
		self.initLegacyCode()

	def initialiseComponents(self):
		# Initialise the components now that the scaffoldInstance is available
		self._initialise_layers()
		self._initialise_cells()
		self._initialise_geometries()
		self._initialise_placement_strategies()
		self._initialise_connection_types()

	def _initialise_cells(self):
		for cell_type in self.configuration.cell_types.values():
			cell_type.initialise(self)

	def _initialise_layers(self):
		for layer in self.configuration.layers.values():
			layer.initialise(self)

	def _initialise_placement_strategies(self):
		for placement in self.configuration.placement_strategies.values():
			placement.initialise(self)

	def _initialise_connection_types(self):
		for connection_type in self.configuration.connection_types.values():
			connection_type.initialise(self)

	def _initialise_geometries(self):
		for geometry in self.configuration.geometries.values():
			geometry.initialise(self)

	def compileNetworkArchitecture(self, tries=1):
		times = np.zeros(tries)
		# Place the cells starting from the lowest density celltypes.
		for i in np.arange(tries, dtype=int):
			t = time.time()
			cell_types = sorted(self.configuration.cell_types.values(), key=lambda x: x.density)
			for cell_type in cell_types:
				cell_type.placement.place(cell_type)
			for connection_type in self.configuration.connection_types.values():
				connection_type.connect()
			times[i] = time.time() - t
			self.save()
			for type in self.configuration.cell_types.values():
				count = self.cells_by_type[type.name].shape[0]
				volume = self.configuration.layers[type.placement.layer].volume
				density_gotten = '%.4g' % (count / volume)
				density_wanted = '%.4g' % (type.placement.get_placement_count(type) / volume)
				percent = int((count / type.placement.get_placement_count(type)) * 100)
				print('{} {} placed ({}%). Desired density: {}. Actual density: {}'.format(count, type.name, percent, density_wanted, density_gotten))
		print('Average runtime: {}'.format(np.average(times)))

	def plotNetworkCache(self):
		plotNetwork(self, from_memory=True)


	def resetNetworkCache(self):
		# Cell positions dictionary per cell type. Columns: X, Y, Z.
		self.cells_by_type = {key: np.empty((0, 5)) for key in self.configuration.cell_types.keys()}
		# Cell positions dictionary per layer. Columns: Type, X, Y, Z.
		self.cells_by_layer = {key: np.empty((0, 5)) for key in self.configuration.layers.keys()}
		# Cells collection. Columns: Cell ID, Type, X, Y, Z.
		self.cells = np.empty((0, 5))
		# Cell connections. Columns: From ID, To ID.
		self.cell_connections = np.empty((0, 2))
		# Cell connections per connection type. Columns: From ID, To ID.
		self.cell_connections_by_type = {}

	def place_cells(self, cell_type, layer, positions):
		# Create an ID for each cell.
		cell_ids = self.allocate_ids(positions.shape[0])
		# Store cells as ID, typeID, X, Y, Z
		cell_data = np.column_stack((
			cell_ids,
			np.ones(positions.shape[0]) * cell_type.id,
			positions
		))
		# Cache them per type
		self.cells_by_type[cell_type.name] = np.concatenate((
			self.cells_by_type[cell_type.name],
			cell_data
		))
		# Cache them per layer
		self.cells_by_layer[layer.name] = np.concatenate((
			self.cells_by_layer[layer.name],
			cell_data
		))
		# Store
		self.cells = np.concatenate((
			self.cells,
			cell_data
		))

	def allocate_ids(self, count):
		IDs = np.array(range(self._nextId, self._nextId + count))
		self._nextId += count
		return IDs

	def connect_cells(self, connection_type, connectome_data, connection_name = None):
		name = connection_name or connection_type.name
		if name in self.cell_connections_by_type:
			cache = self.cell_connections_by_type[name]
			self.cell_connections_by_type[name] = np.concatenate((cache, connectome_data))
		else:
			self.cell_connections_by_type[name] = np.copy(connectome_data)
		# Store all the connections
		self.cell_connections = np.concatenate((self.cell_connections, connectome_data))

	def save(self):
		f = h5py.File('scaffold_new_test.hdf5', 'w')
		celltype_names = self.configuration.cell_type_map
		position_dset = f.create_dataset('positions', data=self.cells)
		position_dset.attrs['types'] = celltype_names
		f.create_group('connections')
		for key, connectome_data in self.cell_connections_by_type.items():
			dset = f['connections'].create_dataset(key, data=connectome_data)
			dset.attrs['name'] = key
			# Maybe from/to information can be stored here aswell.
		f.create_dataset('connectome', data=self.cell_connections)
		f.close()


	def initLegacyCode(self):
		self.placement_stats = {key: {} for key in self.configuration.cell_types.keys()}
		for key, subdic in self.placement_stats.items():
			subdic['number_of_cells'] = []
			subdic['total_n_{}'.format(key)] = 0
			if key != 'purkinje':
				subdic['{}_subl'.format(key)] = 0
