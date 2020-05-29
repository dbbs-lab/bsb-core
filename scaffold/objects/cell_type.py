from .. import config
from ..placement import PlacementStrategy


@config.node
class Representation:
    pass


@config.node
class Plotting:
    pass


@config.node
class CellType:
    name = config.attr(key=True)
    layer = config.attr()
    density = config.attr(type=float)
    planar_density = config.attr(type=float)
    count = config.attr(type=int)

    placement = config.attr(type=PlacementStrategy, required=True)
    representation = config.attr(type=Representation, required=True)
    plotting = config.attr(type=Plotting)
