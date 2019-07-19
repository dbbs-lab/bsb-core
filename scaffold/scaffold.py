from .statistics import Statistics
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.plotNetwork(from_memory=True)

    def plotNetwork(self, file=None, from_memory=False):
        if from_memory:
            plt.interactive(True)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pos = self.final_cell_positions['Granule Cell']
            ax.scatter3D(pos[:,0], pos[:,1], pos[:,2])
            plt.show(block=True)

    def initLegacyCode(self):
        self.final_cell_positions = {key: [] for key in self.configuration.CellTypes.keys()}
        self.placement_stats = {key: {} for key in self.configuration.CellTypes.keys()}
        for key, subdic in self.placement_stats.items():
        	subdic['number_of_cells'] = []
        	subdic['total_n_{}'.format(key)] = 0
        	if key != 'purkinje':
        		subdic['{}_subl'.format(key)] = 0
