from .. import config
from .component import SimulationComponent


@config.node
class ConnectionModel(SimulationComponent):
    tag = config.attr(type=str, key="name")

    def get_connectivity_set(self):
        return self.scaffold.get_connectivity_set(self.tag)
