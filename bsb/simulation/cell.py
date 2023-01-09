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
    Whether this cell type is a relay type. Relay types, during simulation, instantly
    transmit incoming spikes to their targets.
    """
    parameters = config.list(type=Parameter)

    def __lt__(self, other):
        try:
            return self.name < other.name
        except Exception:
            return True
