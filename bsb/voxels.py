import numpy as np
from scipy import ndimage
from time import sleep
import functools


class Voxels:
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
