from .statistics import Statistics
from .plotting import plotNetwork
import numpy as np
import time
from .trees import TreeCollection
from .models import CellType
from .connectivity import ConnectionStrategy

###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads morphologies, morphologies, ...
#    * Creates network architecture
#    * Sets up simulation

class Scaffold:

	def __init__(self, config, from_file=None):
		self.configuration = config
		self.reset_network_cache()
		# Debug statistics, unused.
		self.statistics = Statistics(self)
		self._initialise_output_formatter()
		self.trees = type('scaffold.trees', (object,), {})()
		self.trees.__dict__['cells'] = TreeCollection('cells', self.output_formatter)
		self._nextId = 0
		# Use the configuration to initialise all components such as cells and layers
		# to prepare for the network architecture compilation.
		self.initialiseComponents()
		self.initialiseSimulators()

		# Tell the output formatter that we've loaded from an output and initialise scaffold from it.
		if from_file:
			self.output_formatter.file = from_file
			self.output_formatter.init_scaffold()

	def initialiseComponents(self):
		# Initialise the components now that the scaffoldInstance is available
		self._initialise_layers()
		self._initialise_cells()
		self._initialise_morphologies()
		self._initialise_placement_strategies()
		self._initialise_connection_types()
		self._initialise_simulations()

	def report(self, message, level=2):
		if self.configuration.verbosity >= level:
			print(message)

	def initialiseSimulators(self):
		self.simulators = self.configuration.simulators

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

	def _initialise_morphologies(self):
		for geometry in self.configuration.morphologies.values():
			geometry.initialise(self)

	def _initialise_simulations(self):
		for simulation in self.configuration.simulations.values():
			simulation.initialise(self)
			for sim_cell in simulation.cell_models.values():
				sim_cell.initialise(self)
			for sim_connection in simulation.connection_models.values():
				sim_connection.initialise(self)
			for stimulus in simulation.devices.values():
				stimulus.initialise(self)

	def _initialise_output_formatter(self):
		self.output_formatter = self.configuration.output_formatter
		# Alias the output formatter under its other functions
		self.morphology_repository = self.output_formatter
		self.tree_handler = self.output_formatter
		self.output_formatter.initialise(self)

	def compile_network(self, tries=1):
		times = np.zeros(tries)
		# Place the cells starting from the lowest density cell_types.
		for i in np.arange(tries, dtype=int):
			t = time.time()
			sorted_cell_types = CellType.resolve_order(self.configuration.cell_types)
			for cell_type in sorted_cell_types:
				# Place cell type according to PlacementStrategy
				cell_type.placement.place(cell_type)
				# Construct a tree of the placed cells
				self.trees.cells.create_tree(cell_type.name, self.cells_by_type[cell_type.name][:, 2:5])
			sorted_connection_types = ConnectionStrategy.resolve_order(self.configuration.connection_types)
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
				if self.configuration.verbosity > 1:
					print('{} {} placed ({}%). Desired density: {}. Actual density: {}'.format(count, type.name, percent, density_wanted, density_gotten))
			if self.configuration.verbosity > 1:
				print('Average runtime: {}'.format(np.average(times)))

	def plotNetworkCache(self):
		plotNetwork(self, from_memory=True)


	def reset_network_cache(self):
		# Cell positions dictionary per cell type. Columns: X, Y, Z.
		self.cells_by_type = {key: np.empty((0, 5)) for key in self.configuration.cell_types.keys()}
		# Cell positions dictionary per layer. Columns: Type, X, Y, Z.
		self.cells_by_layer = {key: np.empty((0, 5)) for key in self.configuration.layers.keys()}
		# Cells collection. Columns: Cell ID, Type, X, Y, Z.
		self.cells = np.empty((0, 5))
		# Cell connections per connection type. Columns: From ID, To ID.
		self.cell_connections_by_tag = {}
		self.appends = {}
		self.placement_stitching = []
		self._connectivity_set_meta = {}

	def run_simulation(self, simulation_name):
		if not simulation_name in self.configuration.simulations:
			raise Exception("Unknown simulation '{}', choose from: {}".format(
				simulation_name,
				", ".join(self.configuration.simulations.keys())
			))
		simulation = self.configuration.simulations[simulation_name]
		with self.output_formatter.load() as hdf5:
			simulator = simulation.prepare(hdf5)
		simulation.simulate(simulator)

	def place_cells(self, cell_type, layer, positions):
		cell_count = positions.shape[0]
		if cell_count == 0:
			return
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

		placement_dict = self.statistics.cells_placed
		if not cell_type.name in placement_dict:
			placement_dict[cell_type.name] = 0
		placement_dict[cell_type.name] += cell_count
		if not hasattr(cell_type.placement, 'cells_placed'):
			cell_type.placement.__dict__['cells_placed'] = 0
		cell_type.placement.cells_placed += cell_count
		# Keep track of the order of placement, so that it can be emulated in simulators
		self.placement_stitching.append((cell_type.id, cell_ids[0], cell_count))

	def allocate_ids(self, count):
		IDs = np.array(range(self._nextId, self._nextId + count), dtype=int)
		self._nextId += count
		return IDs

	def connect_cells(self, connection_type, connectome_data, tag=None, meta=None):
		# Allow 1 connection type to store multiple connectivity datasets by utilizing tags
		tag = tag or connection_type.name
		# Keep track of relevant tags in the connection_type object
		if not tag in connection_type.tags:
			connection_type.tags.append(tag)
		# Store the connections based on their tag, and append them if a dataset already exists under the same tag.
		if tag in self.cell_connections_by_tag:
			cache = self.cell_connections_by_tag[tag]
			self.cell_connections_by_tag[tag] = np.concatenate((cache, connectome_data))
		else:
			self.cell_connections_by_tag[tag] = np.copy(connectome_data)
		if not meta is None:
			self._connectivity_set_meta[tag] = meta

	def append_dset(self, name, data):
		self.appends[name] = data

	def get_cells_by_type(self, name):
		if not name in self.cells_by_type or self.cells_by_type[name].shape[0] == 0:
			if self.output_formatter.has_cells_of_type(name):
				if not name in self.configuration.cell_types.keys():
					raise Exception("Attempting to load a cell type '{}' that is present in the output storage, but not in the currently loaded configuration.".format(name))
				self.cells_by_type[name] = self.output_formatter.get_cells_of_type(name)
				return self.cells_by_type[name]
			else:
				raise Exception("Cell type '{}' not found in network cache or output storage".format(name))
		else:
			return self.cells_by_type[name]

	def get_connection_types_by_cell_type(self, postsynaptic=[], presynaptic=[]):
		def any_intersect(l1, l2, f=lambda x: x):
			if not l2: # Return True if there's no pre/post targets specified
				return True
			for e1 in l1:
				if f(e1) in l2:
					return True
			return False

		connection_types = self.configuration.connection_types
		connection_items = connection_types.items()
		filtered_connection_items = list(filter(lambda c:
			any_intersect(c[1].to_cell_types, postsynaptic, lambda x: x.name) and
			any_intersect(c[1].from_cell_types, presynaptic, lambda x: x.name),
			connection_items
		))
		return dict(filtered_connection_items)

	def get_connections_by_cell_type(self, any=None, postsynaptic=None, presynaptic=None):
		if any is None and postsynaptic is None and presynaptic is None:
			raise ArgumentError("No cell types specified")
		# Initialize empty omitted lists
		postsynaptic = postsynaptic if not postsynaptic is None else []
		presynaptic = presynaptic if not presynaptic is None else []
		if not any is None: # Add any cell types as both post and presynaptic targets
			postsynaptic.extend(any)
			presynaptic.extend(any)
		# Find the connection types that have the specified targets
		connection_types = self.get_connection_types_by_cell_type(postsynaptic, presynaptic)
		# Map them to a list of tuples with the 1st element the connection type
		# and the connection matrices appended behind it.
		return list(map(lambda x: (x, *x.get_connection_matrices()), connection_types.values()))

	def save(self):
		self.output_formatter.save()

	def translate_cell_ids(self, data, cell_type):
		return data + self.output_formatter.get_type_map(cell_type)[0]

	def get_connection_type(self, name):
		if not name in self.configuration.connection_types:
			raise Exception("Unknown connection type '{}'".format(name))
		return self.configuration.connection_types[name]

	def get_cell_type(self, name):
		if not name in self.configuration.cell_types:
			raise Exception("Unknown cell type '{}'".format(name))
		return self.configuration.cell_types[name]
