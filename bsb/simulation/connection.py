from .. import config
from .component import SimulationComponent
from ..config import refs


@config.node
class ConnectionModel(SimulationComponent):
    connection_type = config.ref(refs.conn_type_ref, key="name")
