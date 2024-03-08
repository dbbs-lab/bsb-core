import typing

from .. import config
from ..config import refs
from ..config._attrs import cfglist
from .component import SimulationComponent
from .parameter import Parameter

if typing.TYPE_CHECKING:
    from ..cell_types import CellType


@config.node
class CellModel(SimulationComponent):
    """
    Cell models are simulator specific representations of a cell type.
    """

    cell_type: "CellType" = config.ref(refs.cell_type_ref, key="name")
    """
    The cell type that this model represents
    """
    parameters: cfglist[Parameter] = config.list(type=Parameter)
    """
    The parameters of the model.
    """

    def __lt__(self, other):
        try:
            return self.name < other.name
        except Exception:
            return True

    def get_placement_set(self, chunks=None):
        return self.cell_type.get_placement_set(chunks=chunks)


__all__ = ["CellModel"]
