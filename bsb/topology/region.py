"""
Module for the Region types.
"""

import abc
import typing

import numpy as np

from .. import config
from ..config import refs, types
from ..exceptions import ConfigurationError
from ..reporting import warn
from ._layout import Layout

if typing.TYPE_CHECKING:
    from ..core import Scaffold
    from .partition import Partition


@config.dynamic(attr_name="type", required=False, default="group", auto_classmap=True)
class Region(abc.ABC):
    """
    Base region.

    When arranging will simply call arrange/layout on its children but won't cause any
    changes itself.
    """

    scaffold: "Scaffold"

    name: str = config.attr(key=True)
    children: list[typing.Union["Region", "Partition"]] = config.reflist(
        refs.regional_ref, backref="region", required=True
    )
    """Reference to Regions or Partitions belonging to this region."""

    @property
    def data(self):
        # The data property is read-only to users, but `_data` is assigned
        # during the layout process
        return self._data

    def get_dependencies(self):
        return self.children.copy()

    def __boot__(self):
        pass

    def get_layout(self, hint):
        layouts = [dep.get_layout(hint) for dep in self.get_dependencies()]
        return Layout(hint.data.copy(), owner=self, children=layouts)

    def do_layout(self, hint):
        layout = self.get_layout(hint)
        layout.accept()

    @abc.abstractmethod
    def rotate(self, rotation):  # pragma: nocover
        pass

    @abc.abstractmethod
    def translate(self, offset):  # pragma: nocover
        pass

    @abc.abstractmethod
    def scale(self, factors):  # pragma: nocover
        pass


@config.node
class RegionGroup(Region, classmap_entry="group"):
    """
    Base implementation of Region.
    Any transformation on the region will be directly
    applied to its children (Regions or Partitions).
    """

    def rotate(self, rotation):
        for child in self.children:
            child.rotate(rotation)

    def translate(self, offset):
        for child in self.children:
            child.translate(offset)

    def scale(self, factors):
        for child in self.children:
            child.scale(factors)


@config.node
class Stack(RegionGroup, classmap_entry="stack"):
    """
    Stack components on top of each other and adjust its own height accordingly.
    """

    axis: typing.Union[typing.Literal["x"], typing.Literal["y"], typing.Literal["z"]] = (
        config.attr(type=types.in_(["x", "y", "z"]), default="z")
    )
    """Axis along which the stack's children will be stacked"""
    anchor: typing.Union["Region", "Partition"] = config.ref(refs.regional_ref)
    """Reference to one child of the stack, which origin will become the origin of the stack"""

    def _resolve_anchor_offset(self, children, axis_idx):
        """
        Check if the anchor is one of the children of the stack and
        if so, return the offset so the anchor is at the origin of the stack.
        """
        children_owners = [child._owner for child in children]
        if self.anchor is not None and self.anchor in children_owners:
            index = children_owners.index(self.anchor)
            return children[index].data.ldc[axis_idx] - sum(
                children[i].data.dimensions[axis_idx] for i in range(index)
            )
        else:
            # if anchor is not defined or one of the children
            # then the origin of the stack corresponds to the origin of the first child
            return children[0].data.ldc[axis_idx]

    def get_layout(self, hint):
        layout = super().get_layout(hint)
        axis_idx = ("x", "y", "z").index(self.axis)
        trans_eye = np.zeros(3)
        trans_eye[axis_idx] = 1

        cumul_offset = self._resolve_anchor_offset(layout.children, axis_idx)
        for child in layout.children:
            if child.data is None:
                warn(f"Skipped layout arrangement of {child._owner.name} in {self.name}")
                continue
            translation = (
                layout.data.ldc[axis_idx] + cumul_offset - child.data.ldc
            ) * trans_eye
            if not np.allclose(0, translation):
                child.propose_translate(translation)
            cumul_offset += child.data.dimensions[axis_idx]
        ldc = layout.data.ldc
        mdc = layout.data.mdc
        mdc[axis_idx] = ldc[axis_idx] + cumul_offset
        return layout

    def rotate(self, rotation):
        for child in self.children:
            child.rotate(rotation)

    def translate(self, offset):
        for child in self.children:
            child.translate(offset)

    def scale(self, factors):
        for child in self.children:
            child.scale(factors)


__all__ = ["Region", "RegionGroup", "Stack"]
