from .. import config
from .._util import SortableByAfter


@config.node
class SimulationComponent(SortableByAfter):
    name = config.attr(key=True)

    def __boot__(self):
        self.simulation = self._config_parent._config_parent

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of simulation components required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        self.after = []

    def has_after(self):
        return hasattr(self, "after")
