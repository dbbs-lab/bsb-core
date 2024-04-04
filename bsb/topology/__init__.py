"""
Topology module
"""

from ._layout import box_layout
from .partition import AllenStructure, Layer, NrrdVoxels, Partition
from .region import Region, RegionGroup, Stack


def create_topology(regions, ldc, mdc):
    """
    Create a topology from group of regions. Will check for root regions, if there's not
    exactly 1 root region a :class:`~.topology.region.RegionGroup` will be created as new
    root.

    :param regions: Any iterable of regions.
    :type regions: Iterable
    :param ldc: Least dominant corner of the topology. Forms the suggested outer bounds of
      the topology together with the `mdc`.
    :param mdc: Most dominant corner of the topology. Forms the suggested outer bounds of
      the topology together with the `mdc`.
    """
    regions = list(regions)
    if len(roots := get_root_regions(regions)) == 1:
        topology = roots[0]
    else:
        topology = RegionGroup(children=roots, name="topology")
    hint = box_layout(ldc, mdc)
    topology.do_layout(hint)
    return topology


def get_partitions(regions):
    """
    Get all of the partitions belonging to the group of regions and their subregions.

    :param regions: Any iterable of regions.
    :type regions: Iterable
    """

    def collect_deps(region, ignore):
        # get_dependencies can be overridden, so `list` it to avoid mutation of userdata
        deps = list(region.get_dependencies())
        regions = [d for d in deps if not is_partition(d)]
        parts = set(p for p in deps if is_partition(p))
        ignore.update(parts)
        for r in regions:
            # Only check unchecked regions
            if r not in ignore:
                ignore.add(r)
                parts.update(collect_deps(r, ignore))
        return parts

    partitions = set()
    for region in regions:
        partitions.update(p := collect_deps(region, set()))

    return partitions


def is_partition(obj):
    """
    Checks if an object is a partition.
    """
    return (
        hasattr(obj, "get_layout")
        and hasattr(obj, "to_chunks")
        and hasattr(obj, "chunk_to_voxels")
    )


def is_region(obj):
    """
    Checks if an object is a region.
    """
    return hasattr(obj, "get_layout") and hasattr(obj, "do_layout")


def get_root_regions(regions):
    """
    Get all of the root regions, not belonging to any other region in the given group.

    :param regions: Any iterable of regions.
    :type regions: Iterable
    """
    managed = set()

    def collect_deps(region, ignore):
        # get_dependencies can be overridden, so `list` it to avoid mutation of userdata
        deps = list(region.get_dependencies()).copy()
        for dep in deps:
            # Only check unchecked regions, ignore visited & partitions
            if dep not in ignore and is_region(dep):
                ignore.add(dep)
                extra_deps = collect_deps(dep, ignore)
                ignore.update(extra_deps)
        return deps

    # Give `managed` as the mutable ignore arg so that it is filled with all regionals
    # encountered as dependencies by the `collect_deps` recursive function.
    for region in regions:
        collect_deps(region, managed)

    return list(set(list(regions)) - managed)


__all__ = [
    "box_layout",
    "create_topology",
    "get_partitions",
    "get_root_regions",
    "is_partition",
    "is_region",
]
