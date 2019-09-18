import numpy as np
from .morphologies import Morphology as BaseMorphology
from .helpers import ConfigurableClass

class CellType:

    def __init__(self, name, placement=None):
        self.name = name
        self.placement = placement

    def validate(self):
        '''
            Check whether this CellType is valid to be used in the simulation.
        '''
        pass

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.id = scaffoldInstance.configuration.cell_type_map.index(self.name)
        self.validate()

    def set_morphology(self, morphology):
        '''
            Set the Morphology class for this cell type.

            :param morphology: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type morphology: Instance of a subclass of scaffold.morphologies.Morphology
        '''
        if not issubclass(type(morphology), BaseMorphology):
            raise Exception("Only subclasses of scaffold.morphologies.Morphology can be used as cell morphologies.")
        self.morphology = morphology

    def set_placement(self, placement):
        self.placement = placement

class dimensions:
    @property
    def width(self):
        return self.dimensions[0]

    @property
    def height(self):
        return self.dimensions[1]

    @property
    def depth(self):
        return self.dimensions[2]

    @property
    def volume(self):
        return np.prod(self.dimensions)

class origin:
    def X(self):
        return self.origin[0]

    @property
    def Y(self):
        return self.origin[1]

    @property
    def Z(self):
        return self.origin[2]

class Layer(dimensions, origin):

    def __init__(self, name, origin, dimensions, scaling=True):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = np.array(origin)
        # Dimensions in the XYZ axes.
        self.dimensions = np.array(dimensions)
        self.volumeOccupied = 0.
        # Should this layer scale when the simulation volume is resized?
        self.scaling = scaling

    @property
    def available_volume(self):
        return self.volume - self.volumeOccupied

    @property
    def thickness(self):
        return self.dimensions[1]

    def allocateVolume(volume):
        self.volumeOccupied += volume

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance

class NestCell(ConfigurableClass):

    node_name = 'simulations.?.cell_models'
    required = ['parameters']

    def validate(self):
        pass

    def get_parameters(self, model=None):
        # Get the default synapse parameters
        params = self.parameters.copy()
        # If a model is specified, fetch model specific parameters
        if not model is None:
            # Raise an exception if the requested model is not configured.
            if not hasattr(self, model):
                raise Exception("Missing parameters for '{}' model in '{}'".format(model, self.name))
            # Merge in the model specific parameters
            params.update(cell_model.__dict__[model])
        return params

class NestConnection(ConfigurableClass):
    node_name = 'simulations.?.connection_models'

    def validate(self):
        pass
