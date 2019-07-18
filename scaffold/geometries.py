import abc
from .helpers import ConfigurableClass


class Geometry(ConfigurableClass):

    def __init__(self):
        pass

class GranuleCellGeometry(Geometry):

    def validate(self):
        pass

class PurkinjeCellGeometry(Geometry):
    def validate(self):
        pass

class GolgiCellGeometry(Geometry):
    casts = {
        'dendrite_radius': float,
        'axon_x': float,
        'axon_y': float,
        'axon_z': float,
    }

    def validate(self):
        pass

class RadialGeometry(Geometry):
    def validate(self):
        pass

class NoGeometry(Geometry):
    def validate(self):
        pass
