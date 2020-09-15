from .component import SimulationComponent
from .. import config
from ..config import refs


@config.node
class CellModel(SimulationComponent):
    cell_type = config.ref(refs.cell_type_ref, key="name")

    def is_relay(self):
        return self.cell_type.relay

    @property
    def relay(self):
        return self.is_relay()
