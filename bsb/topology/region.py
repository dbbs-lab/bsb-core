"""
Module for the Region types.
"""

from .. import config
from ..config import types, refs
from ..exceptions import *


@config.dynamic(required=False, default="group", auto_classmap=True)
class Region:
    """
    Base region.

    When arranging will simply call arrange/layout on its children but won't cause any
    changes itself.
    """

    name = config.attr(key=True)
    offset = config.attr(
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
        self.boundaries.offset(self.offset)
        for p in self.partitions:
            if hasattr(p, "arrange"):
                p.arrange(self.boundaries)
            else:
                p.layout(self.boundaries)
            if not hasattr(p, "boundaries"):
                raise MissingBoundaryError(
                    f"{p} did not define any boundaries after layout call."
                )


class RegionGroup(Region, classmap_entry="group"):
    origin = None


@config.node
class Stack(Region, classmap_entry="stack"):
    """
    Stack components on top of each other based on their ``stack_index`` and adjust its
    own height accordingly.
    """

    axis = config.attr(default="y")

    def arrange(self, boundary):
        boundary.offset(self.offset)
        stack_height = 0
        for p in sorted(
            self.get_dependencies(), key=lambda p: getattr(p, "stack_index", 0)
        ):
            if hasattr(p, "arrange"):
                p.arrange(boundary.copy())
            else:
                p.layout(boundary.copy())
            if not hasattr(p, "boundaries"):
                raise MissingBoundaryError(
                    f"{p} did not define any boundaries after layout call."
                )
            p.boundaries.y += stack_height
            stack_height += p.boundaries.height
        boundary.height = stack_height
        self.boundaries = boundary
