from ..helpers import ConfigurableClass, SortableByAfter


class SimulationComponent(ConfigurableClass, SortableByAfter):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.simulation = None

    def get_config_node(self):
        return self.node_name + "." + self.name

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of simulation components required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        self.after = []

    def has_after(self):
        return hasattr(self, "after")
