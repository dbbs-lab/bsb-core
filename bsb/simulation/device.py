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

    def get_checkpoints(self, duration, resolution):
        """If checkpoints attribute is not set return an empty list, otherwise return a list of checkpoints (in ms).
        If only a float it is provided it is assumed to be the time interval between checkpoints
        """
        if hasattr(self, "checkpoints"):
            if isinstance(self.checkpoints, float):
                import numpy as np

                multiple = self.checkpoints / resolution
                if multiple != int(multiple):
                    raise ValueError(
                        f"In device {self.name} , Checkpoints must be a multiple of {resolution}"
                    )
                chkp_array = np.delete(np.arange(0, duration, self.checkpoints), 0)
                return chkp_array
            else:
                return self.checkpoints
        else:
            return []


__all__ = ["DeviceModel"]
