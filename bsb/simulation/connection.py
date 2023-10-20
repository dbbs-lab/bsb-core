import typing

from .. import config
from .component import SimulationComponent
from ..config import refs

if typing.TYPE_CHECKING:
    from ..connectivity import ConnectionStrategy


@config.node
class ConnectionModel(SimulationComponent):
    connection_type: "ConnectionStrategy" = config.ref(refs.conn_type_ref, key="name")
