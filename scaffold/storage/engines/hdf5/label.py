from ...interfaces import Label as ILabel
from .resource import Resource
import numpy as np

_root = "/cells/labels/"


class Label(Resource, ILabel):
    def __init__(self, engine, label):
        super().__init__(engine, _root + label)
        self.tag = label

    @classmethod
    def list(cls):
        return Resource(engine, _root).keys()

    @property
    def cells(self):
        if not self.exists():
            return set()
        return set(self.get_dataset())

    def store(self, identifiers):
        if self.exists():
            self.remove()
        self.create(identifiers, dtype=int)

    def label(self, identifiers):
        cells = self.cells
        cells.update(identifiers)
        self.store(list(cells))

    def unlabel(self, identifiers):
        cells = self.cells
        self.store(list(cells - set(identifiers)))
