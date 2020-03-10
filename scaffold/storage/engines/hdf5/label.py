from ...interfaces import Label as ILabel
from .resource import Resource
import numpy as np


class Label(Resource, ILabel):
    def __init__(self, handler, label):
        root = "/cells/labels/"
        super().__init__(handler, root + label)
        self.label = label

    @property
    def cells(self):
        if not self.exists():
            return set()
        return set(self.get_dataset())

    def store(self, identifiers):
        if self.exists():
            self.remove()
        self.create(identifiers, dtype=int)

    def add(self, identifiers):
        cells = self.cells
        cells.update(identifiers)
        self.store(list(cells))
