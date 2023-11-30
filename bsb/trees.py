"""
Module for binary space partitioning, to facilitate optimal runtime complexity for n-point
problems.
"""

import abc

from rtree import index as rtree


class BoxTreeInterface(abc.ABC):
    """
    Tree for fast lookup of queries of axis aligned rhomboids.
    """

    @abc.abstractmethod
    def query(self, boxes, unique=False):
        """
        Should return a generator that yields lists of intersecting IDs per query box if
        ``unique=False``. If ``unique=True``, yield a flat list of unique intersecting box
        IDs for all queried boxes.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class _BoxRTree(BoxTreeInterface):
    """
    Tree for fast lookup of queries of axis aligned rhomboids using the Rtree package.
    """

    def __init__(self, boxes):
        self._rtree = rtree.Index(properties=rtree.Property(dimension=3))
        for id, box in enumerate(boxes):
            self._rtree.insert(id, box)

    def __len__(self):
        return len(self._rtree)

    def query(self, boxes, unique=False):
        """
        Given an iterable of ``(min_x, min_y, min_z, max_x, max_y, max_z)`` box tuples,
        find all the boxes that intersect with them.

        :param boxes: Boxes to look for intersections with.
        :type boxes: Iterable[Tuple[float, float, float, float, float, float]]
        :param unique: If ``True``, return a flat generator of unique results. If ``False``
            (default), per box in ``boxes``, return a list of intersecting boxes.
        :returns: See ``unique``.
        :rtype: Union[Iterator[List[Tuple[float, float, float, float, float, float]]],
            Iterator[Tuple[float, float, float, float, float, float]]]
        """
        all_ = (list(self._rtree.intersection(box, objects=False)) for box in boxes)
        if unique:
            seen = set()
            # Double for loop over results, skipping those that have been seen before.
            yield from (
                seen.add(elem) or elem for arr in all_ for elem in arr if elem not in seen
            )
        else:
            yield from all_


# Cheapo provider pattern.
class BoxTree(_BoxRTree):
    """
    Tree for fast lookup of repeat queries of axis aligned rhomboids.
    """

    pass


__all__ = ["BoxTreeInterface", "BoxTree"]
