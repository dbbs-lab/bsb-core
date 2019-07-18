import abc
from .helpers import ConfigurableClass

class PlacementStrategy(ConfigurableClass):
    @abc.abstractmethod
    def place(self, scaffold, cellType):
        pass

class LayeredRandomWalk(PlacementStrategy):

    def validate(self):
        # Check if the layer is given and exists.
        config = self.scaffold.configuration
        if not hasattr(self, 'layer'):
            raise Exception("Required attribute Layer missing from {}".format(self.name))
        if not self.layer in config.Layers:
            raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))

    def place(self, cellType):
        pass
