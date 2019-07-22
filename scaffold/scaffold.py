from .statistics import Statistics
from .plotting import plotNetwork
import numpy as np
import h5py
from pprint import pprint

###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads geometries, morphologies, ...
#    * Creates network architecture
#    * Sets up simulation

class Scaffold:

	def __init__(self, config):
		self.configuration = config
		# Cell positions dictionary per cell type. Columns: X, Y, Z.
		self.CellsByType = {key: np.empty((0, 3)) for key in config.CellTypes.keys()}
		# Cell positions dictionary per layer. Columns: Type, X, Y, Z.
		self.CellsByLayer = {key: np.empty((0, 4)) for key in config.Layers.keys()}
		# Cell positions dictionary. Columns: Cell ID, Type, X, Y, Z.
		self.Cells = np.empty((0, 5))
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
		self._initialiseLayers()
		self._initialiseCells()
		self._initialisePlacementStrategies()

	def _initialiseCells(self):
		for name, cellType in self.configuration.CellTypes.items():
			cellType.initialise(self)

	def _initialiseLayers(self):
		for name, layer in self.configuration.Layers.items():
			layer.initialise(self)

	def _initialisePlacementStrategies(self):
		for name, placement in self.configuration.PlacementStrategies.items():
			placement.initialise(self)

	def compileNetworkArchitecture(self):
		# Place the cells starting from the lowest density celltypes.
		cellTypes = sorted(self.configuration.CellTypes.values(), key=lambda x: x.density)
		for cellType in cellTypes:
			cellType.placement.place(cellType)

		self.save()
		plotNetwork(self, from_memory=True)

	def placeCells(self, cellType, layer, positions):
		# Store cells per type as X, Y, Z
		self.CellsByType[cellType.name] = np.concatenate((
			self.CellsByType[cellType.name],
			positions
		))
		# Store cells per layer as typeID, X, Y, Z
		positionsWithTypeId = np.column_stack((
			np.ones(positions.shape[0]) * cellType.id,
			positions
		))
		self.CellsByLayer[layer.name] = np.concatenate((
			self.CellsByLayer[layer.name],
			positionsWithTypeId
		))
		# Ask the scaffold for an ID per cell, thread safe?
		CellIDs = self.allocateIDs(positions.shape[0])
		# Store cells as ID, typeID, X, Y, Z
		positionsWithIdAndTypeId = np.column_stack((
			CellIDs,
			positionsWithTypeId
		))
		self.Cells = np.concatenate((
			self.Cells,
			positionsWithIdAndTypeId
		))

	def allocateIDs(self, count):
		IDs = np.array(range(self._nextId, self._nextId + count))
		self._nextId += count
		return IDs

	def save(self):
		f = h5py.File('scaffold_new_test.hdf5', 'w')
		cellTypeIDs = self.configuration.CellTypeIDs
		dset = f.create_dataset('positions', data=self.Cells)
		dset.attrs['types'] = cellTypeIDs
		f.close()


	def initLegacyCode(self):
		self.placement_stats = {key: {} for key in self.configuration.CellTypes.keys()}
		for key, subdic in self.placement_stats.items():
			subdic['number_of_cells'] = []
			subdic['total_n_{}'.format(key)] = 0
			if key != 'purkinje':
				subdic['{}_subl'.format(key)] = 0
