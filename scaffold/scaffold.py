from .statistics import Statistics
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
    	scaffoldInstance.initialiseComponents()

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
