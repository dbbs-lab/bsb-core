from .component import SimulationComponent
from .parameter import Parameter
from .. import config
from ..config import refs


@config.node
class CellModel(SimulationComponent):
    """
    Cell models are simulator specific representations of a cell type.
    """

    cell_type = config.ref(refs.cell_type_ref, key="name")
    """
    The cell type that this model represents
    """
    parameters = config.list(type=Parameter)
    """
    The parameters of the model.
    """

    def __lt__(self, other):
        try:
            return self.name < other.name
        except Exception:
            return True

    def get_placement_set(self):
        return self.cell_type.get_placement_set()
