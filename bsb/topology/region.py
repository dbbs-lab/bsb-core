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
    Stack components on top of each other based on their ``stack_index`` and adjust its
    own height accordingly.
    """

    axis: typing.Union[typing.Literal["x"], typing.Literal["y"], typing.Literal["z"]] = (
        config.attr(type=types.in_(["x", "y", "z"]), default="z")
    )

    def boot(self):
        # Check that layers have different stack indexes and that the indexes are in range
        indexes = np.array(
            [
                child.stack_index
                for child in self.children
                if hasattr(child, "stack_index")
            ]
        )
        if np.any(indexes < 0) or np.any(indexes >= len(self.children)):
            raise ConfigurationError(
                "Layers stack indexes must be in the range of the number of layers"
            )
        if len(indexes) != np.unique(indexes).size:
            raise ConfigurationError("Layers stack indexes must be unique")

    def _resolve_order(self):
        # Sort the children position in the stack. Layers can specify their position within.
        indexes = [
            (child.stack_index if hasattr(child, "stack_index") else -1)
            for child in self.children
        ]
        order = np.zeros_like(indexes)
        current = 0
        for i in range(len(self.children)):
            if i in indexes:
                order[i] = indexes.index(i)
            else:
                order[i] = np.where(np.array(indexes) == -1)[0][current]
                current += 1
        return order

    def get_layout(self, hint):
        layout = super().get_layout(hint)
        axis_idx = ("x", "y", "z").index(self.axis)
        trans_eye = np.zeros(3)
        trans_eye[axis_idx] = 1

        children = np.array(layout.children)[self._resolve_order()]
        # origin of stack corresponds to the origin of the first child
        cumul_offset = children[0].data.ldc[axis_idx]
        for child in children:
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
