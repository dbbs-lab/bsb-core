import numpy as np
from .morphologies import Morphology as BaseMorphology
from .helpers import ConfigurableClass, dimensions, origin

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

    def get_placed_count(self):
        return self.scaffold.statistics.cells_placed[self.name]

    def get_ids(self):
        return np.array(self.scaffold.cells_by_type[self.name][:,0], dtype=int)

    def list_all_morphologies(self):
        if not hasattr(self.morphology, "detailed_morphologies"):
            return []
        morphology_config = self.morphology.detailed_morphologies
        if 'names' in morphology_config:
            m_names = morphology_config['names']
            return m_names
        else:
            raise NotImplementedError("Detailed morphologies can currently only be selected by name.")

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
