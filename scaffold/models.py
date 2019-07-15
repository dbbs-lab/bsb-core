import numpy as np
from .geometries import Geometry as BaseGeometry

class CellType:

    def __init__(self, name, density = 0.):
        self.name = name
        self.density = density
        self.color = '#000000'
        self.geometry = None
        self.morphology = None

    def validate(self):
        '''
            Check whether this CellType is valid to be used in the simulation.
        '''
        if self.geometry == None && self.morphology == None:
            raise Exception("No Geometry or Morphology set for cell type {}".format(self.name))
        if self.placement == None:
            raise Exception("No PlacementStrategy set for cell type {}".format(self.name))
        return true


class GeometricCellType(CellType):

    def __init__(self, name, geometry):
        CellType.__init__(self, name)
        self.setGeometry(geometry)

    def setGeometry(self, geometry):
        '''
            Set the Geometry class for this cell type.

            :param geometry: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type geometry: Instance of a subclass of scaffold.geometries.Geometry
        '''
        if not issubclass(type(geometry), BaseGeometry):
            raise Exception("Only subclasses of scaffold.geometries.Geometry can be used as cell geometries.")
        self.geometry = geometry

class MorphologicCellType(CellType):

    def setMorphology(self, morphology):
        self.morphology = morphology

class Layer:

    def __init__(self, name, origin, dimensions):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = origin
        # Dimensions in the XYZ axes.
        self.dimensions = dimensions

    @property
    def volume(self):
        return np.prod(self.dimensions)
