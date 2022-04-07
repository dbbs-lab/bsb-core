from rtree import index as rtree
import abc


class BoxTreeInterface(abc.ABC):
    @abc.abstractmethod
    def query(self, boxes):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class BoxRTree(BoxTreeInterface):
    def __init__(self, boxes):
        self._rtree = rtree.Index(properties=rtree.Property(dimension=3))
        for id, box in enumerate(boxes):
            self._rtree.insert(id, box)

    def __len__(self):
        return self._rtree.get_size()

    def query(self, boxes, unique=False):
        all_ = ([*self._rtree.intersection(box, objects=False)] for box in boxes)
        if unique:
            seen = set()
            yield from (seen.add(elem) or elem for a in all_ for elem in a if elem not in seen)
        else:
            yield from all_


# Cheapo provider
class BoxTree(BoxRTree):
    pass
