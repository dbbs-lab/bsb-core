import numpy as np
from .geometries import Geometry as BaseGeometry

class CellType:

    def __init__(self, name, density=0., radius=0., ratio=None, ratioTo=None, placement=None):
        self.name = name
        self.density = density
        self.planarDensity = None
        self.radius = radius
        self.color = '#000000'
        self.ratio = ratio
        self.ratioTo = ratioTo
        self.geometry = None
        self.morphology = None
        self.placement = None

    def validate(self):
        '''
            Check whether this CellType is valid to be used in the simulation.
        '''
        if self.geometry == None and self.morphology == None:
            raise Exception("No Geometry or Morphology set for cell type '{}'".format(self.name))
        if self.placement == None:
            raise Exception("No PlacementStrategy set for cell type '{}'".format(self.name))

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.id = scaffoldInstance.configuration.cell_type_map.index(self.name)
        self.validate()

    def set_geometry(self, geometry):
        '''
            Set the Geometry class for this cell type.

            :param geometry: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type geometry: Instance of a subclass of scaffold.geometries.Geometry
        '''
        if not issubclass(type(geometry), BaseGeometry):
            raise Exception("Only subclasses of scaffold.geometries.Geometry can be used as cell geometries.")
        self.geometry = geometry

    def setMorphology(self, morphology):
        self.morphology = morphology

    def setPlacementStrategy(self, placement):
        self.placement = placement

class dimensions:
    @property
    def X(self):
        return self.dimensions[0]

    @property
    def Y(self):
        return self.dimensions[1]

    @property
    def Z(self):
        return self.dimensions[2]

    @property
    def volume(self):
        return np.prod(self.dimensions)

class Layer(dimensions):

    def __init__(self, name, origin, dimensions):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = np.array(origin)
        # Dimensions in the XYZ axes.
        self.dimensions = np.array(dimensions)
        self.volumeOccupied = 0.

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
