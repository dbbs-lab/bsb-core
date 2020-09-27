from .. import config
from ..helpers import SortableByAfter


@config.node
class SimulationComponent(SortableByAfter):
    name = config.attr(key=True)

    def __init__(self, parent=None):
        if parent is None:
            self.adapter = None
        else:
            # Get the parent of the dict  that we are defined in (cell_models,
            # connections_models, device_models, ...). This grandparent is the adapter
            self.adapter = parent._config_parent
        self.simulation = None

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of simulation components required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        self.after = []

    def has_after(self):
        return hasattr(self, "after")
