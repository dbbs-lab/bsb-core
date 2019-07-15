###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads geometries, morphologies, ...
#    * Can be configured
#    * Creates network architecture
#    * Sets up simulation

class Scaffold:

    def __init__(self, config):
        self.configuration = config
