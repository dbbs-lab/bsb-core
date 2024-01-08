import abc

import numpy as np


class Layout:
    """
    Container class for all types of partition data. The layout swaps the data of the
    partition with temporary layout associated data, and tries out experimental changes
    to the partition data, if the layout process fails, the original partition data is
    reinstated.
    """

    def __init__(self, data, owner=None, children=None, frozen=False):
        if children is None:
            children = []
        self._data = data
        self._owner = owner
        self._children = children
        self._frozen = frozen

    @property
    def data(self):
        return self._data

    @property
    def children(self):
        return self._children

    def copy(self):
        return Layout(
            data=self._data.copy(),
            owner=self._owner,
            children=self._children,
            frozen=self._frozen,
        )

    def accept(self):
        self.swap()

    def __getattr__(self, attr):
        if attr.startswith("propose_"):
            f = getattr(self._owner, attr[8:])

            def swapped_execute(*args, **kwargs):
                self.swap()
                f(*args, **kwargs)
                self.swap()

            return swapped_execute

        super().__getattribute__(attr)

    def swap(self):
        if self._owner is not None:
            old = getattr(self._owner, "_data", None)
            self._owner._data = self._data
            self._data = old
        for child in self._children:
            child.swap()


class PartitionData(abc.ABC):
    """
    The partition data is a class that stores the description of a partition for a
    partition. This allows the Partition interface to define mutating operations such as
    translate, rotate, scale; for a dry-run we only have to swap out the actual data with
    temporary data, and the mutation is prevented.
    """

    @abc.abstractmethod
    def copy(self):
        pass


class RhomboidData(PartitionData):
    def __init__(self, ldc, mdc):
        # Least dominant corner
        self.ldc = np.array(ldc, dtype=float, copy=False)
        # Most dominant corner
        self.mdc = np.array(mdc, dtype=float, copy=False)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ldc}, {self.mdc})"

    def copy(self):
        """
        Copy this boundary to a new instance.
        """
        return self.__class__(self.ldc.copy(), self.mdc.copy())

    @property
    def x(self):
        return self.ldc[0]

    @x.setter
    def x(self, value):
        self.mdc[0] += value - self.ldc[0]
        self.ldc[0] = value

    @property
    def y(self):
        return self.ldc[1]

    @y.setter
    def y(self, value):
        self.mdc[1] += value - self.ldc[1]
        self.ldc[1] = value

    @property
    def z(self):
        return self.ldc[2]

    @z.setter
    def z(self, value):
        self.mdc[2] += value - self.ldc[2]
        self.ldc[2] = value

    @property
    def dimensions(self):
        return self.mdc - self.ldc

    @property
    def width(self):
        return self.mdc[0] - self.ldc[0]

    @width.setter
    def width(self, value):
        self.mdc[0] = self.ldc[0] + value

    @property
    def height(self):
        return self.mdc[2] - self.ldc[2]

    @height.setter
    def height(self, value):
        self.mdc[2] = self.ldc[2] + value

    @property
    def depth(self):
        return self.mdc[1] - self.ldc[1]

    @depth.setter
    def depth(self, value):
        self.mdc[1] = self.ldc[1] + value


def box_layout(ldc, mdc):
    return Layout(RhomboidData(ldc, mdc), frozen=True)
