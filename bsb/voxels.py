from . import config
from .config import types
from .trees import BoxTree
import numpy as np
import functools
import abc
import nrrd


class VoxelSet:
    def __init__(self, voxels, size, voxel_data=None, irregular=False):
        voxel_size = np.array(size, copy=False)
        if not len(voxel_size.shape):
            self._cubic = True
        if len(voxel_size.shape) > 1:
            # Voxels given in spatial coords with individual size
            self._sizes = voxel_size
            self._coords = np.array(voxels, copy=False)
            self._regular = False
        elif irregular:
            # Voxels given in spatial coords but of equal size
            self._size = voxel_size
            self._coords = np.array(voxels, copy=False)
            self._regular = False
        else:
            # Voxels given in index coords
            self._size = voxel_size
            self._indices = np.array(voxels, copy=False, dtype=int)
            self._regular = True
        self._voxel_data = voxel_data

    def __iter__(self):
        return iter(self.raw(copy=False))

    def __len__(self):
        return len(self.raw(copy=False))

    def __getitem__(self, index):
        voxels = self.raw(copy=False)[index]
        if self.of_equal_size:
            voxel_size = self._size
        else:
            voxel_size = self._sizes[index]
        if self.has_data:
            voxel_data = self._voxel_data[index]
        else:
            voxel_data = None
        return VoxelSet(voxels, voxel_size, voxel_data)

    def __bool__(self):
        return not self.is_empty

    @property
    def is_empty(self):
        return not len(self)

    @property
    def has_data(self):
        return self._voxel_data is not None

    @property
    def regular(self):
        return self._regular

    @property
    def of_equal_size(self):
        # One size fits all
        return hasattr(self, "_size")

    @property
    def size(self):
        return self.get_size()

    @property
    def data(self):
        return self.get_data()

    @property
    @functools.cache
    def bounds(self):
        return (
            np.min(self.as_spatial_coords(copy=False), axis=0),
            np.max(self.as_spatial_coords(copy=False), axis=0),
        )

    @classmethod
    def empty(cls, size=None):
        return cls(np.empty((0, 3)), np.empty((0, 3)))

    @classmethod
    def one(cls, ldc, mdc, data=None):
        voxels = cls(np.array([ldc]), np.array([mdc - ldc]))
        if data is not None:
            voxels._voxel_data = np.array([data], dtype=object)
        return voxels

    @classmethod
    def concatenate(cls, *sets):
        if (
            any(s.has_data for s in sets)
            or any(not s.regular for s in sets)
            or any(not np.allclose(s._size, sets[0]._size) for s in sets)
        ):
            raise NotImplementedError(
                "Can only concat same size index coords without data for now"
            )
        alloc = np.empty(
            (
                sum(s.raw(copy=False).shape[0] for s in sets),
                sets[0].raw(copy=False).shape[1],
            )
        )
        ptr = 0
        for s in sets:
            l = len(s)
            alloc[ptr : (ptr + l)] = s.raw(copy=False)
        return VoxelSet(alloc, sets[0]._size)

    def raw(self, copy=True):
        coords = self._indices if self.regular else self._coords
        if copy:
            coords = coords.copy()
        return coords

    def get_data(self, copy=True):
        if self.has_data:
            return np.array(self._voxel_data, copy=copy)
        else:
            return np.empty(len(self), dtype=object)

    def get_size(self, copy=True):
        if self.of_equal_size:
            return np.array(self._size, copy=copy)
        else:
            return np.array(self._sizes, copy=copy)

    def get_size_matrix(self, copy=True):
        if self.of_equal_size:
            size = np.ones(3) * self._size
            sizes = np.tile(size, len(self.raw(copy=False)))
        else:
            sizes = self._sizes
            if copy:
                sizes = sizes.copy()
        return sizes

    def as_spatial_coords(self, copy=True):
        if self.regular:
            coords = self._to_spatial_coords()
        else:
            coords = self._coords
            if copy:
                coords = coords.copy()
        return coords

    def as_boxtree(self, cache=False):
        if cache:
            return self._boxtree_cache()
        else:
            return self._boxtree()

    def snap_to_grid(self, grid_size, unique=False):
        if self.regular:
            print(
                "regular voxels",
                self._indices,
                grid_size,
                self._size,
                (grid_size / self._size),
            )
            grid = self._indices // (grid_size / self._size)
            print("outcome", grid)
        else:
            print("irregular voxels", self._coords, grid_size)
            grid = self._coords // grid_size
        voxel_data = self._voxel_data
        if unique:
            if self.has_data:
                grid, id = np.unique(grid, return_index=True, axis=0)
                voxel_data = voxel_data[id]
            else:
                grid = np.unique(grid, axis=0)
        return VoxelSet(grid, grid_size, voxel_data)

    def select(self, ldc, mdc):
        voxel_data = self._voxel_data
        coords = self.as_spatial_coords(copy=False)
        print("Looking for", ldc, mdc, "with voxelset in", self.bounds)
        inside = np.all(np.logical_and(ldc <= coords, coords < mdc), axis=1)
        print(sum(inside))
        return self[inside]

    def select_chunk(self, chunk):
        return self.select(chunk.ldc, chunk.mdc)

    def unique(self):
        raise NotImplementedError("and another one")

    def _to_spatial_coords(self):
        return self._indices * self._size

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
            tiled = coords + np.ones(3) * self._size
            return np.column_stack((coords, tiled))

    @classmethod
    def from_morphology(cls, morphology, estimate_n, with_data=True):
        # Find a good distribution of amount of voxels per side
        size = morphology.meta["mdc"] - morphology.meta["ldc"]
        per_side = _eq_sides(size, estimate_n)
        voxel_size = size / per_side
        branch_vcs = [
            b.as_matrix(with_radius=False) // voxel_size for b in morphology.branches
        ]
        if with_data:
            voxel_reduce = {}
            for branch, point_vcs in enumerate(branch_vcs):
                for point, point_vc in enumerate(point_vcs):
                    voxel_reduce.setdefault(tuple(point_vc), []).append((branch, point))
            voxels = np.array(tuple(voxel_reduce.keys()))
            voxel_data = np.array(list(voxel_reduce.values()), dtype=object)
            return cls(voxels, voxel_size, voxel_data=voxel_data)
        else:
            voxels = np.array(set((itertools.chain.from_iterable(branch_vcs))))
            return cls(voxels, voxel_size)


@config.dynamic(
    attr_name="type", auto_classmap=True, required=True, type=types.in_classmap()
)
class VoxelLoader(abc.ABC):
    @abc.abstractmethod
    def get_voxelset(self):
        pass


@config.node
class NrrdVoxelLoader(VoxelLoader, classmap_entry="nrrd"):
    source = config.attr(type=str, required=True)
    mask_value = config.attr(type=int, required=True)
    voxel_size = config.attr(type=types.voxel_size(), required=True)

    def get_voxelset(self):
        data, header = nrrd.read(self.source)
        voxels = np.transpose(np.nonzero(data == self.mask_value))
        return VoxelSet(voxels, self.voxel_size)


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
