"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..placement import PlacementStrategy
from ..helpers import SortableByAfter


@config.dynamic(attr_name="selector", auto_classmap=True)
class MorphologySelector:
    names = config.list(type=str)


@config.node
class Representation:
    radius = config.attr(type=float, required=True)
    geometrical = config.dict(type=types.any())
    morphological = config.dict(type=MorphologySelector)


@config.node
class Plotting:
    display_name = config.attr()
    color = config.attr()
    opacity = config.attr(type=types.fraction(), default=1.0)


def _not_an_entity(section):
    return "entity" not in section or not bool(section["entity"])


@config.node
class CellType(SortableByAfter):
    name = config.attr(key=True)
    placement = config.attr(type=PlacementStrategy, required=True)
    spatial = config.attr(
        type=Representation, required=_not_an_entity, default={"radius": None}
    )
    plotting = config.attr(type=Plotting)
    relay = config.attr(type=bool, default=False)
    entity = config.attr(type=bool, default=False)

    def get_placement_set(self):
        return self.scaffold.get_placement_set(self.name)

    @classmethod
    def get_ordered(cls, objects):
        return sorted(objects.values(), key=lambda x: x.placement.get_placement_count())

    def has_after(self):
        return hasattr(self.placement, "after")

    def get_after(self):
        return None if not self.has_after() else self.placement.after

    def create_after(self):
        self.placement.after = []
