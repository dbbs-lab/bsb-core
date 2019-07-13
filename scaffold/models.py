

class CellType:

    def __init__(self, name):
        self.name = name

class Layer:

    def __init__(self, name, origin, dimensions):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = origin
        # Dimensions in the XYZ axes.
        self.dimensions = dimensions
