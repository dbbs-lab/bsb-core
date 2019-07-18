import abc
from helpers import CastsConfigurationValues


class Geometry(abc.ABC, CastsConfigurationValues):

    def __init__(self):
        pass

class GranuleCellGeometry(Geometry):
    pass

class PurkinjeCellGeometry(Geometry):
    pass

class GolgiCellGeometry(Geometry):
    pass

class RadialGeometry(Geometry):
    pass

class NoGeometry(Geometry):
    pass
