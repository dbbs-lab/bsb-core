"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..placement import PlacementStrategy
from ..placement.indicator import PlacementIndications
from ..helpers import SortableByAfter


@config.dynamic(
    attr_name="selector",
    auto_classmap=True,
    required=False,
    default="by_name",
)
class MorphologySelector:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is MorphologySelector:
            return
        if not hasattr(cls, "pick"):
            raise RuntimeError("MorphologySelectors must define a `pick` method.")


@config.node
class NameSelector(MorphologySelector, classmap_entry="by_name"):
    names = config.list(type=str)

    def pick(self, morphology):
        return morphology.get_meta()["name"] in self.names


@config.node
class Representation(PlacementIndications):
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
class CellType:
    name = config.attr(key=True)
    spatial = config.attr(
        type=Representation, required=_not_an_entity, default={"radius": None}
    )
    plotting = config.attr(type=Plotting)
    relay = config.attr(type=bool, default=False)
    entity = config.attr(type=bool, default=False)

    def get_placement_set(self):
        return self.scaffold.get_placement_set(self.name)
