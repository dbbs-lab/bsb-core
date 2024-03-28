"""
    Module for the CellType configuration node and its dependencies.
"""

import typing

from . import config
from ._util import obj_str_insert
from .config import types
from .placement.indicator import PlacementIndications

if typing.TYPE_CHECKING:
    from .core import Scaffold


@config.node
class Plotting:
    scaffold: "Scaffold"

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

    scaffold: "Scaffold"

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
    entity = config.attr(type=bool, default=False)
    """
    Whether this cell type is an entity type. Entity types don't have representations in
    space, but can still be connected and simulated.
    """

    def __boot__(self):
        storage = self.scaffold.storage
        if storage.supports("PlacementSet"):
            storage.require_placement_set(self)

    def __lt__(self, other):
        try:
            return self.name < other.name
        except Exception:
            return True

    @obj_str_insert
    def __repr__(self):
        try:
            placements = len(self.get_placement())
        except Exception:
            placements = "?"
        try:
            cells_placed = len(self.get_placement_set())
        except Exception:
            cells_placed = 0
        return f"'{self.name}', {cells_placed} cells, {placements} placement strategies"

    def get_placement(self):
        """
        Get the placement components this cell type is a part of.
        """
        return self.scaffold.get_placement_of(self)

    def get_placement_set(self, *args, **kwargs):
        """
        Retrieve this cell type's placement data

        :param chunks: When given, restricts the placement data to these chunks.
        :type chunks: List[bsb.storage._chunks.Chunk]
        """
        return self.scaffold.get_placement_set(self, *args, **kwargs)

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
            if self is conn_set.pre_type or self is conn_set.post_type:
                conn_set.clear()

    # This property was mostly added so that accidental assignment to `ct.morphologies`
    # instead of `ct.spatial.morphologies` raises an error.
    @property
    def morphologies(self):
        return self.get_morphologies()

    @morphologies.setter
    def morphologies(self, value):
        raise AttributeError(
            "`cell_type.morphologies` is a readonly attribute. Did you mean"
            " `cell_type.spatial.morphologies`?"
        )


__all__ = ["CellType", "PlacementIndications", "Plotting"]
