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
        self.statistics = Statistics(self)
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

    def placeCell(self, cellType, position):
        pass

    def save(self):
        f = h5py.File('scaffold_new_test.hdf5', 'w')
        cellTypeIDs = self.configuration.CellTypeIDs
        data = np.empty((0, 5))
        startingID = 0
        for cellTypeName, cellPositions in self.final_cell_positions.items():
            nCellsOfType = cellPositions.shape[0]
            cellTypeID = cellTypeIDs.index(cellTypeName)
            cellTypeColumn = cellTypeID * np.ones(nCellsOfType)
            endingID = startingID + nCellsOfType
            cellIDColumn = np.array(range(startingID, endingID))
            startingID = endingID
            newData = np.column_stack(( # Create 5 column layout: ID, type, X, Y, Z
                cellIDColumn,
                cellTypeColumn,
                cellPositions
            ))
            data = np.concatenate((
                data,
                newData
            ))
        dset = f.create_dataset('positions', data=data)
        dset.attrs['types'] = cellTypeIDs
        f.close()


    def initLegacyCode(self):
        self.final_cell_positions = {key: [] for key in self.configuration.CellTypes.keys()}
        self.placement_stats = {key: {} for key in self.configuration.CellTypes.keys()}
        for key, subdic in self.placement_stats.items():
        	subdic['number_of_cells'] = []
        	subdic['total_n_{}'.format(key)] = 0
        	if key != 'purkinje':
        		subdic['{}_subl'.format(key)] = 0
