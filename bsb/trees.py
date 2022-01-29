from rtree import index as rtree
import abc


class BoxTreeInterface(abc.ABC):
    @abc.abstractmethod
    def query(self, boxes):
        pass


class BoxRTree(BoxTreeInterface):
    def __init__(self, boxes):
        self._rtree = rtree.Index(properties=rtree.Property(dimension=3))
        for id, box in enumerate(boxes):
            self._rtree.insert(id, box)

    def query(self, boxes):
        return ([*self._rtree.intersection(box, objects=False)] for box in boxes)


# Cheapo provider
class BoxTree(BoxRTree):
    pass
