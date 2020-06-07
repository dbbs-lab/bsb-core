"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..placement import PlacementStrategy


@config.node
class Representation:
    radius = config.attr(type=float, required=True)
    entity = config.attr(type=bool, default=False)
    geometry = config.dict(type=types.any())


@config.node
class Plotting:
    display_name = config.attr()
    color = config.attr()
    opacity = config.attr(type=types.fraction(), default=1.0)


@config.node
class CellType:
    name = config.attr(key=True)
    placement = config.attr(type=PlacementStrategy, required=True)
    spatial = config.attr(type=Representation, required=True)
    plotting = config.attr(type=Plotting)
    relay = config.attr(type=bool, default=False)
