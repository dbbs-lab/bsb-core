from . import config
from .config import types
import numpy as np
import functools
import abc
import nrrd


class VoxelSet:
    def __init__(self, voxels, size, voxel_data=None, irregular=False):
        voxel_size = np.array(size, copy=False)
        if len(size.shape) > 1:
            # Voxels given in spatial coords with individual size
            self._sizes = size
            self._coords = voxels
        elif irregular:
            # Voxels given in spatial coords but of equal size
            self._size = size
            self._coords = voxels
        else:
            # Voxels given in index coords
            self._size = size
            self._indices = voxels
        self._voxel_data = voxel_data

    @property
    def have_data(self):
        return self._voxel_data is not None

    def as_spatial_coords(self, copy=True):
        if hasattr(self, "_coords"):
            coords = self._coords
        else:
            coords = self._to_spatial_coords()
        return coords.copy() if copy else coords

    def as_index_coords(self, copy=True):
        if hasattr(self, "_coords"):
            coords = self._coords
        else:
            coords = self._to_spatial_coords()
        return coords.copy() if copy else coords

    def as_boxtree(self, cache=False):
        if cache:
            return self._boxtree_cache()
        else:
            return self._boxtree()

    @functools.cache
    def _boxtree_cache(self):
        return self._boxtree()

    def _boxtree(self):
        return BoxTree(self.as_boxes())

    def as_boxes(self, cache=False):
        if cache:
            return self._boxes_cache()
        else:
            return self._boxes()

    @functools.cache
    def _boxes_cache(self):
        return self._boxes()

    def _boxes(self):
        coords = self.as_spatial_coords(copy=False)
        if hasattr(self, "_sizes"):
            return np.column_stack((coords, self._sizes))
        else:
            tiled = np.broadcast(self._size, tocorrectshape)
            return np.column_stack((coords, tiled))

    @classmethod
    def from_morphology(cls, morphology, estimate_n):
        # Find a good distribution of amount of voxels per side
        size = morphology.meta["mdc"] - morphology.meta["ldc"]
        per_side = _eq_sides(size, estimate_n)
        voxel_size = size / per_side
        branch_vcs = [
            b.as_matrix(with_radius=False) // voxel_size for b in morphology.branches
        ]
        voxel_reduce = {}
        for branch, point_vcs in enumerate(branch_vcs):
            for point, point_vc in enumerate(point_vcs):
                voxel_reduce.setdefault(tuple(point_vc), []).append((branch, point))
        voxels = np.array(tuple(voxel_reduce.keys()))
        voxel_data = list(voxel_reduce.values())
        return cls(voxels, voxel_size, voxel_data=voxel_data)


def _eq_sides(sides, n):
    # Use the relative magnitudes of each side
    norm = sides / max(sides)
    # Find out how many divisions each side should to form a grid with `n` rhomboids.
    per_side = norm * (n / np.product(norm)) ** (1 / len(sides))
    # Divisions should be integers, and minimum 1
    int_sides = np.maximum(np.floor(per_side), 1)
    order = np.argsort(sides)
    smallest = order[0]
    if len(sides) > 2:
        # Because of the integer rounding the product isn't necesarily optimal, so we keep
        # the safest (smallest) value, and solve the problem again in 1 less dimension.
        solved = int_sides[smallest]
        look_for = n / solved
        others = sides[order[1:]]
        int_sides[order[1:]] = _eq_sides(others, look_for)
    else:
        # In the final 2-dimensional case the remainder of the division is rounded off
        # to the nearest integer, giving the smallest error on the product and final
        # number of rhomboids in the grid.
        largest = order[1]
        int_sides[largest] = round(n / int_sides[smallest])
    return int_sides
