"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..placement import PlacementStrategy


@config.node
class Representation:
    radius = config.attr(type=float, required=True)


@config.node
class Plotting:
    pass


@config.node
class CellType:
    name = config.attr(key=True)

    placement = config.attr(type=PlacementStrategy, required=True)
    spatial = config.attr(type=Representation, required=True)
    plotting = config.attr(type=Plotting)
