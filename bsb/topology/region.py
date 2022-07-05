"""
Module for the Region types.
"""

from ._layout import Layout, RhomboidData
from .. import config
from ..config import types, refs
from ..exceptions import *
from ..reporting import warn
import numpy as np
import abc


@config.dynamic(attr_name="type", required=False, default="group", auto_classmap=True)
class Region(abc.ABC):
    """
    Base region.

    When arranging will simply call arrange/layout on its children but won't cause any
    changes itself.
    """

    name = config.attr(key=True)
    children = config.reflist(refs.regional_ref, backref="region", required=True)

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

    axis = config.attr(default="y")

    def get_layout(self, hint):
        layout = super().get_layout(hint)
        stack_size = 0
        axis_idx = ("x", "y", "z").index(self.axis)
        trans_eye = np.zeros(3)
        trans_eye[axis_idx] = 1

        for child in layout.children:
            if child.data is None:
                warn(f"Skipped layout arrangement of {child._owner.name} in {self.name}")
                continue
            translation = (
                layout.data.ldc[axis_idx] + stack_size - child.data.ldc
            ) * trans_eye
            if not np.allclose(0, translation):
                child.propose_translate(translation)
            stack_size += child.data.dimensions[axis_idx]
        ldc = layout.data.ldc
        mdc = layout.data.mdc
        mdc[axis_idx] = ldc[axis_idx] + stack_size
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
