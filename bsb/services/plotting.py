from ._util import MockModule
from ..exceptions import DependencyError as _DepErr


class PlottingService:
    def __init__(self):
        self.module = MockModule("bsb_plotting")

    def is_mocked(self):
        return hasattr(self.module, "_mocked")

    def plot_network(self, network):
        if self.is_mocked():
            raise _DepErr(
                "Please install the plotting dependencies with `pip install bsb[plot]`"
            )
        return self._module.plot_network(network)
