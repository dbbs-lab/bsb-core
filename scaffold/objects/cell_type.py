from .. import config


@config.node
class CellType:
    name = config.attr(key=True)
    layer = config.attr()
    density = config.attr(type=float)
    planar_density = config.attr(type=float)
    count = config.attr(type=int)
