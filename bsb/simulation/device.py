from .. import config
from ..services import MPI
from .component import SimulationComponent


@config.node
class DeviceModel(SimulationComponent):
    pass
