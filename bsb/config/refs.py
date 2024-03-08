"""
    This module contains shorthand ``reference`` definitions. References are used in the
    configuration module to point to other locations in the Configuration object.

    Minimally a reference is a function that takes the configuration root and the current
    node as arguments, and returns another node in the configuration object::

      def some_reference(root, here):
          return root.other.place

    More advanced usage of references will include custom reference errors.
"""


class Reference:  # pragma: nocover
    def __call__(self, root, here):
        return here

    def up(self, here, to=None):
        if to is None:
            return here._config_parent
        while not isinstance(here, to):
            try:
                here = here._config_parent
            except AttributeError:
                return None
        return here


class CellTypeReference(Reference):
    def __call__(self, root, here):
        return root.cell_types

    def is_ref(self, value):
        from ..cell_types import CellType

        return isinstance(value, CellType)


class ConnectionTypeReference(Reference):
    def __call__(self, root, here):
        return root.connectivity

    def is_ref(self, value):
        from ..connectivity import ConnectionStrategy

        return isinstance(value, ConnectionStrategy)


class PartitionReference(Reference):
    def __call__(self, root, here):
        return root.partitions

    def is_ref(self, value):
        from ..topology import Partition

        return isinstance(value, Partition)


class RegionReference(Reference):
    def __call__(self, root, here):
        return root.regions

    def is_ref(self, value):
        from ..topology import Region

        return isinstance(value, Region)


class RegionalReference(Reference):
    def __call__(self, root, here):
        merged = root.regions.copy()
        merged.update(root.partitions)
        return merged

    def is_ref(self, value):
        from ..topology import Partition, Region

        return isinstance(value, Region) or isinstance(value, Partition)


class PlacementReference(Reference):
    def __call__(self, root, here):
        return root.placement

    def is_ref(self, value):
        from ..placement import PlacementStrategy

        return isinstance(value, PlacementStrategy)


class ConnectivityReference(Reference):
    def __call__(self, root, here):
        return root.connectivity

    def is_ref(self, value):
        from ..connectivity import ConnectionStrategy

        return isinstance(value, ConnectionStrategy)


class SimCellModelReference(Reference):
    def __call__(self, root, here):
        from ..simulation.simulation import Simulation

        sim = self.up(here, Simulation)
        return sim.cell_models

    def is_ref(self, value):
        from ..simulation.cell import CellModel

        return isinstance(value, CellModel)


cell_type_ref = CellTypeReference()
conn_type_ref = ConnectionTypeReference()
partition_ref = PartitionReference()
placement_ref = PlacementReference()
connectivity_ref = ConnectivityReference()
regional_ref = RegionalReference()
region_ref = RegionReference()
sim_cell_model_ref = SimCellModelReference()

__all__ = [
    "Reference",
    "cell_type_ref",
    "conn_type_ref",
    "partition_ref",
    "placement_ref",
    "connectivity_ref",
    "regional_ref",
    "region_ref",
    "sim_cell_model_ref",
]
__api__ = ["Reference"]
