"""
Topology module
"""

from .region import Region
from .partition import Partition
import numpy as np


def create_topology(regions):
    regions = list(regions)
    if len(roots := get_root_regions(regions)) == 1:
        topology = roots[0]
    else:
        topology = Region(cls="group", partitions=regions, name="topology")
        topology._partitions = regions
    return topology


def get_partitions(regions):
    def collect_deps(region, ignore):
        # get_dependencies can be overridden, so `list` it to avoid mutation of userdata
        deps = list(region.get_dependencies())
        regions = [d for d in deps if hasattr(d, "arrange")]
        parts = set(p for p in deps if hasattr(p, "layout"))
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


def get_root_regions(regions):
    managed = set()

    def collect_deps(region, ignore):
        # get_dependencies can be overridden, so `list` it to avoid mutation of userdata
        deps = list(region.get_dependencies()).copy()
        for dep in deps:
            # Only check unchecked regions, ignore visited & partitions
            if dep not in ignore and hasattr(dep, "arrange"):
                ignore.add(dep)
                extra_deps = collect_deps(dep, ignore)
                ignore.update(extra_deps)
        return deps

    # Give `managed` as the mutable ignore arg so that it is filled with all regionals
    # encountered as dependencies.
    for region in regions:
        collect_deps(region, managed)

    return list(set(list(regions)) - managed)


class Boundary:
    def __init__(self, ldc, mdc):
        # Least dominant corner
        self.ldc = np.array(ldc)
        # Most dominant corner
        self.mdc = np.array(mdc)

    def copy(self):
        return self.__class__(self.ldc, self.mdc)

    @property
    def dimensions(self):
        return self.mdc - self.ldc

    # Make the point properties
    for _i, _name in enumerate(("x", "y", "z")):

        def _get(self, n=_i):
            return self.ldc[n]

        def _set(self, v, n=_i):
            self.mdc[n] += v - self.ldc[n]
            self.ldc[n] = v

        vars()[_name] = property(_get).setter(_set)

    # Make the dimension properties
    for _i, _name in enumerate(("width", "height", "depth")):

        def _get(self, n=_i):
            return self.mdc[n] - self.ldc[n]

        def _set(self, v, n=_i):
            self.mdc[n] = self.ldc[n] + v

        vars()[_name] = property(_get).setter(_set)

    # Cleanup stray variables
    del vars()["_i"]
    del vars()["_name"]
    del vars()["_get"]
    del vars()["_set"]


class BoxBoundary(Boundary):
    def __init__(self, point, dimensions, centered=False):
        point = np.array(point)
        dimensions = np.array(dimensions)
        self.centered = centered
        # Shift by half the dimensions if the cube is centered around its point, if not
        # use the point as LDC and point + dims as MDC
        ldc = point - (centered * dimensions / 2)
        mdc = point + (dimensions / (1 + centered))
        super().__init__(ldc, mdc)

    @property
    def point(self):
        return self.ldc + (self.centered * self.dimensions / 2)

    def copy(self):
        return self.__class__(self.point, self.dimensions, self.centered)
