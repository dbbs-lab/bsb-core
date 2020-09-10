"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..placement import PlacementStrategy


@config.node
class Representation:
    radius = config.attr(type=float, required=True)
    geometry = config.dict(type=types.any())


@config.node
class Plotting:
    display_name = config.attr()
    color = config.attr()
    opacity = config.attr(type=types.fraction(), default=1.0)


def _not_an_entity(section):
    return "entity" not in section or not bool(section["entity"])


@config.node
class CellType:
    name = config.attr(key=True)
    placement = config.attr(type=PlacementStrategy, required=True)
    spatial = config.attr(
        type=Representation, required=_not_an_entity, default={"radius": None}
    )
    plotting = config.attr(type=Plotting)
    relay = config.attr(type=bool, default=False)
    entity = config.attr(type=bool, default=False)
