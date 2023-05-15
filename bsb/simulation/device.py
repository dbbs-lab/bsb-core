from .. import config
from ..services import MPI
from .component import SimulationComponent


@config.node
class DeviceModel(SimulationComponent):
    def implement(self, adapter, simdata):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )
