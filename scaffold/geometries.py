import abc
from .helpers import CastsConfigurationValues


class Geometry(abc.ABC, CastsConfigurationValues):

    def __init__(self):
        pass

class GranuleCellGeometry(Geometry):
    pass

class PurkinjeCellGeometry(Geometry):
    pass

class GolgiCellGeometry(Geometry):
    casts = {
        'dendrite_radius': float,
        'axon_x': float,
        'axon_y': float,
        'axon_z': float,
    }

class RadialGeometry(Geometry):
    pass

class NoGeometry(Geometry):
    pass
