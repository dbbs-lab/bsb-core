from .. import config
from .._util import obj_str_insert


@config.node
class SimulationComponent:
    name = config.attr(key=True)

    @property
    def simulation(self):
        return self._config_parent._config_parent

    @obj_str_insert
    def __str__(self):
        return f"'{self.name}'"
