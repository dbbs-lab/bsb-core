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

    def initialiseComponents(self):
        # Initialise the components now that the scaffoldInstance is available
        self._initialiseLayers()
        self._initialiseCells()

    def _initialiseCells(self):
        for name, cellType in self.configuration.CellTypes.items():
            cellType.initialise(self)

    def _initialiseLayers(self):
        for name, layer in self.configuration.Layers.items():
            layer.initialise(self)
