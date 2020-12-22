"""
Module for the Region onfiguration nodes.
"""

from .. import config
from ..config import types, refs


@config.dynamic(required=False, default="y_stack", auto_classmap=True)
class Region:
    name = config.attr(key=True)
    origin = config.attr(
        type=types.list(type=float, size=3),
        default=lambda: [0.0, 0.0, 0.0],
        call_default=True,
    )
    partitions = config.reflist(refs.regional_ref)

    def get_dependencies(self):
        if self.partitions:
            return self.partitions.copy()
        else:
            return []

    def __boot__(self):
        pass

    def arrange(self, boundaries):
        self.boundaries = boundaries
        for p in self.partitions:
            if hasattr(p, "arrange"):
                p.arrange(self.boundaries)
            else:
                p.layout(self.boundaries)


class RegionGroup(Region, classmap_entry="group"):
    origin = None


@config.node
class YStack(Region, classmap_entry="y_stack"):
    def arrange(self, boundary):
        stack_height = 0
        for p in self.partitions:
            if hasattr(p, "arrange"):
                child_boundaries = p.arrange(boundary.copy())
            else:
                child_boundaries = p.layout(boundary.copy())
            child_boundaries.y += stack_height
            stack_height += child_boundaries.height
        boundary.height = stack_height
        self.boundary = boundary
        return boundary
