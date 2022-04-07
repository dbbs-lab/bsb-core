"""
    Module for the CellType configuration node and its dependencies.
"""

from . import config
from .config import types
from .placement import PlacementStrategy
from .placement.indicator import PlacementIndications
from .helpers import SortableByAfter
from .exceptions import *
import abc


@config.node
class Plotting:
    display_name = config.attr()
    color = config.attr()
    opacity = config.attr(type=types.fraction(), default=1.0)


def _not_an_entity(section):
    return "entity" not in section or not bool(section["entity"])


@config.node
class CellType:
    """
    Information on a population of cells.
    """

    name = config.attr(key=True)
    spatial = config.attr(
        type=PlacementIndications, required=_not_an_entity, default={"radius": None}
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
