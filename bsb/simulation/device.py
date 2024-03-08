from .. import config
from .component import SimulationComponent


@config.node
class DeviceModel(SimulationComponent):
    def implement(self, adapter, simulation, simdata):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )


__all__ = ["DeviceModel"]
