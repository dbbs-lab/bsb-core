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
    """
    Label used to display this cell type in plots.
    """
    color = config.attr()
    """
    Color used to display this cell type in plots.
    """
    opacity = config.attr(type=types.fraction(), default=1.0)
    """
    Opacity used to display this cell type in plots.
    """


def _not_an_entity(section):
    return "entity" not in section or not bool(section["entity"])


@config.node
class CellType:
    """
    Information on a population of cells.
    """

    name = config.attr(key=True)
    """
    Name of the cell type, equivalent to the key it occurs under in the configuration.
    """
    spatial = config.attr(
        type=PlacementIndications, required=_not_an_entity, default={"radius": None}
    )
    """
    Spatial information about the cell type such as radius and density, and geometric or
    morphological information.
    """
    plotting = config.attr(type=Plotting)
    """
    Plotting information about the cell type, such as color and labels.
    """
    relay = config.attr(type=bool, default=False)
    """
    Whether this cell type is a relay type. Relay types, during simulation, instantly
    transmit incoming spikes to their targets.
    """
    entity = config.attr(type=bool, default=False)
    """
    Whether this cell type is an entity type. Entity types don't have representations in
    space, but can still be connected and simulated.
    """

    def __boot__(self):
        storage = self.scaffold.storage
        storage._PlacementSet.require(storage._engine, self)

    def get_placement_set(self, chunks=None):
        """
        Retrieve this cell type's placement data

        :param chunks: When given, restricts the placement data to these chunks.
        :type chunks: List[bsb.storage.Chunk]
        """
        return self.scaffold.get_placement_set(self.name, chunks=chunks)

    def get_morphologies(self):
        """
        Return the list of morphologies of this cell type.

        :rtype: List[~bsb.storage.interfaces.StoredMorphology]
        """
        if "morphologies" not in self.spatial:
            return []
        else:
            return self.scaffold.storage.morphologies.select(*self.spatial.morphologies)

    def clear(self):
        """
        Clear all the placement and connectivity data associated with this cell type.
        """
        self.clear_placement()
        self.clear_connections()

    def clear_placement(self):
        """
        Clear all the placement data associated with this cell type. Connectivity data
        will remain, but be invalid.
        """
        self.get_placement_set().clear()

    def clear_connections(self):
        """
        Clear all the connectivity data associated with this cell type. Any connectivity
        set that this cell type is a part of will be entirely removed.
        """
        for conn_set in self.scaffold.get_connectivity_sets():
            if self is conn_set.presynaptic or self is conn_set.postsynaptic:
                conn_set.clear()
