from .component import SimulationComponent
from .. import config
from ..config import refs


@config.node
class CellModel(SimulationComponent):
    """
    Cell models are simulator specific representations of a cell type.
    """

    cell_type = config.ref(refs.cell_type_ref, key="name")
    """
    The cell type associated to this cell_model.
    """
    relay = config.attr(type=bool, default=False)
    """
    Whether this cell type is a relay type. Relay types, during simulation, instantly
    transmit incoming spikes to their targets.
    """
