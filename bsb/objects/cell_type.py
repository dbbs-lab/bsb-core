"""
    Module for the CellType configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..placement import PlacementStrategy
from ..placement.indicator import PlacementIndications
from ..helpers import SortableByAfter
from ..exceptions import *
import abc


@config.dynamic(
    attr_name="selector",
    auto_classmap=True,
    required=False,
    default="by_name",
)
class MorphologySelector(abc.ABC):
    @abc.abstractmethod
    def validate(self, all_morphos):
        pass

    @abc.abstractmethod
    def pick(self, morphology):
        pass


@config.node
class NameSelector(MorphologySelector, classmap_entry="by_name"):
    names = config.list(type=str)

    def validate(self, all_morphos):
        missing = set(self.names) - {m.get_meta()["name"] for m in all_morphos}
        if missing:
            raise MissingMorphologyError(
                f"Morphology repository misses the following morphologies required by {self._config_parent._config_parent.get_node_name()}: {', '.join(missing)}"
            )

    def pick(self, morphology):
        return morphology.get_meta()["name"] in self.names


@config.node
class Representation(PlacementIndications):
    geometrical = config.dict(type=types.any())
    morphological = config.list(type=MorphologySelector)


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

    def get_placement_set(self, chunks=None):
        return self.scaffold.get_placement_set(self.name, chunks=chunks)

    def clear(self, force=False):
        self.clear_placement(force=force)
        self.clear_connections(force=force)

    def clear_placement(self, force=False):
        self.get_placement_set().clear()

    def clear_connections(self, force=False):
        for conn_set in self.scaffold.get_connectivity_sets():
            if self is conn_set.presynaptic or self is conn_set.postsynaptic:
                conn_set.clear()
