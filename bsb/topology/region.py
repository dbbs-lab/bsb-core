"""
Module for the Region types.
"""

from ._layout import Layout
from .. import config
from ..config import types, refs
from ..exceptions import *
import abc


@config.dynamic(attr_name="type", required=False, default="group", auto_classmap=True)
class Region(abc.ABC):
    """
    Base region.

    When arranging will simply call arrange/layout on its children but won't cause any
    changes itself.
    """

    name = config.attr(key=True)
    children = config.reflist(refs.regional_ref, required=True)

    def get_dependencies(self):
        if self.children:
            return self.children.copy()
        else:
            return []

    def __boot__(self):
        pass

    def get_layout(self, hint):
        layouts = [dep.get_layout(hint) for dep in self.get_dependencies()]
        return Layout(None, owner=self, children=layouts)

    def do_layout(self, hint):
        layout = self.get_layout(hint)
        layout.accept()

    @abc.abstractmethod
    def rotate(self, rotation):
        pass

    @abc.abstractmethod
    def translate(self, offset):
        pass

    @abc.abstractmethod
    def scale(self, factors):
        pass


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
class Stack(Region, classmap_entry="stack"):
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
            translation = (hint.ldc[axis_idx] + stack_size - child.data.ldc) * trans_eye
            if not np.allclose(0, translation):
                child.propose_translate(translation)
            stack_size += child.data.dimensions * trans_eye
        ldc = hint.ldc.copy()
        mdc = hint.mdc
        mdc[axis_idx] = ldc[axis_idx] + stack_size
        layout.data = RhomboidData(ldc, mdc)
        return layout

    def do_layout(self, hint):
        layout = self.get_layout(hint)

    def rotate(self, rotation):
        for child in self.children:
            child.rotate(rotation)

    def translate(self, offset):
        for child in self.children:
            child.translate(offset)

    def scale(self, factors):
        for child in self.children:
            child.scale(factors)
