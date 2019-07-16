import abc


class Geometry(abc.ABC):

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
