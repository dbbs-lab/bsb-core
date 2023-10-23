import typing

from .. import config
from .component import SimulationComponent

if typing.TYPE_CHECKING:
    from ..connectivity import ConnectionStrategy


@config.node
class ConnectionModel(SimulationComponent):
    tag: str = config.attr(type=str, key=True)

    def get_connectivity_set(self):
        return self.scaffold.get_connectivity_set(self.tag)
