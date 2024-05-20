import functools
import itertools

import numpy as np

from .exceptions import EmptyVoxelSetError
from .trees import BoxTree


class VoxelData(np.ndarray):
    """
    Chunk identifier, consisting of chunk coordinates and size.
    """

    def __new__(cls, data, keys=None):
        if data.ndim < 2:
            return super().__new__(np.ndarray, data.shape, dtype=object)
        obj = super().__new__(cls, data.shape, dtype=object)
        obj[:] = data
        if keys is not None:
            keys = [str(k) for k in keys]
            if len(set(keys)) != len(keys):
                raise ValueError("Data keys must be unique")
            if len(keys) != data.shape[1]:
                raise ValueError("Amount of data keys must match amount of data columns")
            obj._keys = keys
        else:
            obj._keys = []
        return obj

    def __getitem__(self, index):
        index, keys = self._rewrite_index(index)
        vd = super().__getitem__(index)
        if isinstance(vd, VoxelData):
            if len(keys) > 0 and len(vd) != vd.size / len(keys):
                vd = vd.reshape(-1, len(keys))
            vd._keys = keys
        return vd

    def __array_finalize__(self, obj):
        if obj is not None:
            self._keys = []

    @property
    def keys(self):
        """
        Returns the keys, or column labels, associated to each data column.
        """
        return self._keys.copy()

    def copy(self):
        """
        Return a new copy of the voxel data
        """
        new = super().copy()
        new._keys = self._keys.copy()
        return new

    def _split_index(self, index):
        try:
            if isinstance(index, tuple):
                cols = [self._keys.index(idx) for idx in index if isinstance(idx, str)]
                keys = [self._keys[c] for c in cols]
                index = tuple(idx for idx in index if not isinstance(idx, str))
            elif isinstance(index, str):
                cols = [self._keys.index(index)]
                keys = [self._keys[cols[0]]]
                index = (slice(None),)
            else:
                index = (index,)
                cols = None
                keys = getattr(self, "_keys", [])
        except ValueError as e:
            key = str(e).split("'")[1]
            raise IndexError(f"Voxel data key '{key}' does not exist.") from None
        return index, cols, keys

    def _rewrite_index(self, index):
        index, cols, keys = self._split_index(index)
        if cols:
            return (*index, cols), keys
        else:
            return index, keys


class VoxelSet:
    def __init__(self, voxels, size, data=None, data_keys=None, irregular=False):
        """
        Constructs a voxel set from the given voxel indices or coordinates.

        :param voxels: The spatial coordinates, or grid indices of each voxel. Nx3 matrix
        :type voxels: numpy.ndarray
        :param size: The global or individual size of the voxels. If it has 2 dimensions
          it needs to have the same length as `voxels`, and will be used as individual
          voxels.
        :type size: numpy.ndarray
        :param data:

        .. warning::

            If :class:`numpy.ndarray` are passed, they will not be copied in order to save
            memory and time. You may accidentally change a voxelset if you later change
            the same array.
        """
        voxels = np.array(voxels, copy=False)
        voxel_size = np.array(size, copy=False)
        if voxels.dtype.name == "object":
            raise ValueError("Couldn't convert given `voxels` to a voxel matrix")
        if voxels.ndim != 2:
            if not len(voxels):
                # Try some massaging in case of empty arrays
                voxels = voxels.reshape(-1, 3)
                if not len(voxel_size):
                    voxel_size = voxel_size.reshape(-1, 3)
            else:
                raise ValueError("`voxels` needs to be convertable to a 2D matrix")
        if voxels.ndim == 2 and voxels.shape[1] != 3:
            raise ValueError("`voxels` needs to have 3 columns, 1 for each spatial dim.")
        if not _is_broadcastable(voxels.shape, voxel_size.shape):
            raise ValueError(
                f"Shape {voxel_size.shape} of `size` is"
                + f" invalid for voxel shape {voxels.shape}"
            )
        if data is not None:
            if isinstance(data, VoxelData):
                if data_keys is None:
                    self._data = data
                else:
                    self._data = VoxelData(data, keys=data_keys)
            else:
                data = np.array(data, copy=False)
                if data.ndim < 2:
                    cols = len(data_keys) if data_keys else 1
                    data = data.reshape(-1, cols)
                self._data = VoxelData(data, keys=data_keys)
            if len(self._data) != len(voxels):
                raise ValueError("`voxels` and `data` length unequal.")
        else:
            self._data = None

        if not len(voxel_size.shape):
            self._cubic = True
        if voxel_size.ndim > 1:
            if voxel_size.size != voxels.size:
                raise ValueError("`voxels` and `size` length unequal.")
            # Voxels given in spatial coords with individual size
            self._sizes = voxel_size
            self._coords = voxels
            self._regular = False
        elif irregular:
            # Voxels given in spatial coords but of equal size
            self._size = voxel_size
            self._coords = voxels
            self._regular = False
        else:
            # Voxels given in index coords
            self._size = voxel_size
            self._indices = np.array(voxels, copy=False, dtype=int)
            self._regular = True

    def __iter__(self):
        return iter(self.get_raw(copy=False))

    def __array__(self):
        return self.get_raw(copy=False)

    def __len__(self):
        return len(self.get_raw(copy=False))

    def __getitem__(self, index):
        if self.has_data:
            data = self._data[index]
            index, _, _ = self._data._split_index(index)
        else:
            data, _ = None, None
        if isinstance(index, tuple) and len(index) > 1:
            raise IndexError("Too many indices for VoxelSet, maximum 1.")
        voxels = self.get_raw(copy=False)[index]
        if self._single_size:
            voxel_size = self._size.copy()
        else:
            voxel_size = self._sizes[index]
        if voxels.ndim == 1:
            voxels = voxels.reshape(-1, 3)
        return VoxelSet(voxels, voxel_size, data)

    def __getattr__(self, key):
        if key in self._data._keys:
            return self.get_data(key)
        else:
            return super().__getattribute__(key)

    def __str__(self):
        cls = type(self)
        obj = f"<{cls.__module__}.{cls.__name__} object at {hex(id(self))}>"
        if self.is_empty:
            insert = "[EMPTY] "
        else:
            insert = f"with {len(self)} voxels from "
            insert += f"{tuple(self.bounds[0])} to {tuple(self.bounds[1])}, "
            if self.regular:
                insert += f"same size {self.size}, "
            else:
                insert += "individual sizes, "
            if self.has_data:
                if self._data.keys:
                    insert += f"with keyed data ({', '.join(self._data.keys)}) "
                else:
                    insert += f"with {self._data.shape[-1]} data columns "
            else:
                insert += "without data "
        return obj.replace("at 0x", insert + "at 0x")

    __repr__ = __str__

    def __bool__(self):
        return not self.is_empty

    @property
    def is_empty(self):
        """
        Whether the set contain any voxels

        :rtype: bool
        """
        return not len(self)

    @property
    def has_data(self):
        """
        Whether the set has any data associated to the voxels

        :rtype: bool
        """
        return self._data is not None

    @property
    def regular(self):
        """
        Whether the voxels are placed on a regular grid.

        :rtype: bool
        """
        return self._regular

    @property
    def of_equal_size(self):
        return self._single_size or len(np.unique(self._sizes, axis=0)) < 2

    @property
    def equilateral(self):
        """
        Whether all sides of all voxels have the same lengths.

        :rtype: bool
        """
        return np.unique(self.get_size(copy=False)).shape == (1,)

    @property
    def size(self):
        """
        The size of the voxels. When it is 0D or 1D it counts as the size for all voxels,
        if it is 2D it is 1 an individual size per voxel.

        :rtype: numpy.ndarray
        """
        return self.get_size()

    @property
    def volume(self):
        if self._single_size:
            voxel_volume = np.abs(np.prod(self.get_size(copy=False) * np.ones(3)))
            return voxel_volume * len(self)
        else:
            return np.sum(np.abs(np.prod(self.get_size_matrix(copy=False), axis=1)))

    @property
    def data(self):
        """
        The size of the voxels. When it is 0D or 1D it counts as the size for all voxels,
        if it is 2D it is 1 an individual size per voxel.

        :rtype: Union[numpy.ndarray, None]
        """
        return self.get_data()

    @property
    def data_keys(self):
        return self._data.keys

    @property
    def raw(self):
        return self.get_raw()

    @property
    @functools.cache
    def bounds(self):
        """
        The minimum and maximum coordinates of this set.

        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        if self.is_empty:
            raise EmptyVoxelSetError("Empty VoxelSet has no bounds.")
        boxes = self.as_boxes()
        dims = boxes.shape[1] // 2
        return (
            np.min(boxes[:, :dims], axis=0),
            np.max(boxes[:, dims:], axis=0),
        )

    @classmethod
    def empty(cls, size=None):
        return cls(np.empty((0, 3)), np.empty((0, 3)))

    @classmethod
    def one(cls, ldc, mdc, data=None):
        ldc = np.array(ldc, copy=False).reshape(-1)
        mdc = np.array(mdc, copy=False).reshape(-1)
        if ldc.shape != (3,) or mdc.shape != (3,):
            raise ValueError(
                "Arguments to `VoxelSet.one` should be shape (3,) minmax coords of voxel."
            )
        return cls(np.array([ldc]), np.array([mdc - ldc]), np.array([data]))

    @classmethod
    def concatenate(cls, *sets):
        # Short circuit "stupid" concat requests
        if not sets:
            return cls.empty()
        elif len(sets) == 1:
            return sets[0].copy()

        primer = None
        # Check which sets we are concatenating, maybe we can keep them in reduced data
        # forms. If they don't line up, we expand and concatenate the expanded forms.
        if any(
            # `primer` is assigned the first non-empty set, all sizes must match sizes can
            # still be 0D, 1D or 2D, but if they're allclose broadcasted it is fine! :)
            not np.allclose(s.get_size(copy=False), primer.get_size(copy=False))
            for s in sets
            if s and (primer := primer or s)
        ):
            sizes = primer.get_size()
            if len(sizes.shape) > 1:
                # We happened to pick a VoxelSet that has a size matrix of equal sizes,
                # so we take the opportunity to reduce it.
                sizes = sizes[0]
            if len(sizes.shape) > 0 and np.allclose(sizes, sizes[0]):
                # Voxelset is actually even cubic regular!
                sizes = sizes[0]
            if all(s.regular for s in sets):
                # Index coords with same sizes can simply be stacked
                voxels = np.concatenate([s.get_raw(copy=False) for s in sets])
                irregular = False
            else:
                voxels = np.concatenate([s.as_spatial_coords(copy=False) for s in sets])
                irregular = True
        else:
            # We can't keep a single size, so expand into a matrix where needed and concat
            sizes = np.concatenate([s.get_size_matrix(copy=False) for s in sets])
            voxels = np.concatenate([s.as_spatial_coords(copy=False) for s in sets])
            irregular = True

        if any(s.has_data for s in sets):
            fillers = [s.get_data(copy=False) for s in sets]
            # Find all keys among data to concatenate
            all_keys = set(itertools.chain(*(f.keys for f in fillers if f is not None)))
            # Create an index for each key
            keys = [*sorted(set(all_keys), key=str)]
            # Allocate enough columns for all keys, or a data array with more unlabelled
            # columns than that.
            md = max(len(keys), *(f.shape[1] for f in fillers if f is not None))
            if not keys:
                keys = None
            elif md > len(keys):
                # Find and pad `keys` with labels for the extra numerical columns.
                extra = md - len(keys)
                new_nums = (s for c in itertools.count() if (s := str(c)) not in keys)
                keys.extend(itertools.islice(new_nums, extra))
                keys.extend(range(len(keys), md))
                keys = sorted(keys)
            ln = [len(s) for s in sets]
            data = np.empty((sum(ln), md), dtype=object)
            ptr = 0
            for len_, fill in zip(ln, fillers):
                if fill is not None:
                    if not fill.keys:
                        cols = slice(None, fill.shape[1])
                    else:
                        cols = [keys.index(key) for key in fill.keys]
                    data[ptr : (ptr + len_), cols] = fill
                    ptr += len_
        else:
            data = None
            keys = None
        return VoxelSet(voxels, sizes, data=data, data_keys=keys, irregular=irregular)

    def copy(self):
        if self.is_empty:
            return VoxelSet.empty()
        else:
            return VoxelSet(
                self.raw,
                self.get_size(copy=True),
                self.get_data(copy=True) if self.has_data else None,
                irregular=not self.regular,
            )

    def get_raw(self, copy=True):
        coords = self._indices if self.regular else self._coords
        if copy:
            coords = coords.copy()
        return coords

    def get_data(self, index=None, /, copy=True):
        if self.has_data:
            if index is not None:
                return self._data[index]
            else:
                return self._data.copy()
        else:
            return None

    def get_size(self, copy=True):
        if self._single_size:
            return np.array(self._size, copy=copy)
        else:
            return np.array(self._sizes, copy=copy)

    def get_size_matrix(self, copy=True):
        if self._single_size:
            size = np.ones(3) * self._size
            sizes = np.tile(size, (len(self.get_raw(copy=False)), 1))
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

    def as_boxes(self, cache=False):
        if cache:
            return self._boxes_cache()
        else:
            return self._boxes()

    def as_boxtree(self, cache=False):
        if cache:
            return self._boxtree_cache()
        else:
            return self._boxtree()

    def snap_to_grid(self, voxel_size, unique=False):
        if self.regular:
            grid = self._indices // _squash_zero(voxel_size / _squash_zero(self._size))
        else:
            grid = self._coords // _squash_zero(voxel_size)
        data = self._data
        if unique:
            if self.has_data:
                grid, id = np.unique(grid, return_index=True, axis=0)
                data = data[id]
            else:
                grid = np.unique(grid, axis=0)
        return VoxelSet(grid, voxel_size, data)

    def resize(self, size):
        val = np.array(size, copy=False)
        if val.dtype.name == "object":
            raise ValueError("Size must be number type")
        if val.ndim > 1:
            if len(val) != len(self):
                raise ValueError("Individual voxel sizes must match amount of voxels.")
            if self.regular:
                self._coords = self.as_spatial_coords()
                del self._indices
                self._regular = False
        self._size = size

    def crop(self, ldc, mdc):
        coords = self.as_spatial_coords(copy=False)
        inside = np.all(np.logical_and(ldc <= coords, coords < mdc), axis=1)
        return self[inside]

    def crop_chunk(self, chunk):
        return self.crop(chunk.ldc, chunk.mdc)

    @classmethod
    def fill(cls, positions, voxel_size, unique=True):
        return cls(positions, 0, irregular=True).snap_to_grid(voxel_size, unique=unique)

    def coordinates_of(self, positions):
        if not self.regular:
            raise ValueError("Cannot find a unique voxel index in irregular VoxelSet.")
        return positions // self.get_size()

    def index_of(self, positions):
        coords = self.coordinates_of(positions)
        map_ = {tuple(vox_coord): i for i, vox_coord in enumerate(self)}
        return np.array([map_.get(tuple(coord), np.nan) for coord in coords])

    def inside(self, positions):
        mask = np.zeros(len(positions), dtype=bool)
        ldc, mdc = self._box_bounds()
        for voxel in zip(ldc, mdc):
            mask |= np.all((positions >= ldc) & (positions < mdc), axis=1)
        return mask

    def unique(self):
        raise NotImplementedError("and another one")

    @property
    def _single_size(self):
        # One size fits all
        return hasattr(self, "_size")

    def _to_spatial_coords(self):
        return self._indices * self._size

    @functools.cache
    def _boxtree_cache(self):
        return self._boxtree()

    def _boxtree(self):
        return BoxTree(self.as_boxes())

    @functools.cache
    def _boxes_cache(self):
        return self._boxes()

    def _box_bounds(self):
        base = self.as_spatial_coords(copy=False)
        sizes = self.get_size(copy=False)
        shifted = base + sizes
        lt0 = sizes < 0
        if np.any(lt0):
            mdc = np.where(lt0, base, shifted)
            ldc = np.where(lt0, shifted, base)
        else:
            ldc = base
            mdc = shifted
        return ldc, mdc

    def _boxes(self):
        return np.column_stack(self._box_bounds())

    @classmethod
    def from_morphology(cls, morphology, estimate_n, with_data=True):
        meta = morphology.meta
        if "mdc" in meta and "ldc" in meta:
            ldc, mdc = meta["ldc"], meta["mdc"]
        else:
            ldc, mdc = morphology.bounds
        # Find a good distribution of amount of voxels per side
        size = mdc - ldc
        per_side = _eq_sides(size, estimate_n)
        voxel_size = size / per_side
        _squash_temp = _squash_zero(voxel_size)
        branch_vcs = [b.points // _squash_temp for b in morphology.branches]
        if with_data:
            voxel_reduce = {}
            for branch, point_vcs in enumerate(branch_vcs):
                for point, point_vc in enumerate(point_vcs):
                    voxel_reduce.setdefault(tuple(point_vc), []).append((branch, point))
            voxels = np.array(tuple(voxel_reduce.keys()))
            # Transfer the voxel data into an object array
            voxel_data_data = tuple(voxel_reduce.values())
            # We need a bit of a workaround so that numpy doesn't make a regular from the
            # `voxel_data_data` list of lists, when it has a matrix shape.
            voxel_data = np.empty(len(voxel_data_data), dtype=object)
            for i in range(len(voxel_data_data)):
                voxel_data[i] = voxel_data_data[i]
            return cls(voxels, voxel_size, data=voxel_data)
        else:
            voxels = np.unique(np.concatenate(branch_vcs), axis=0)
            return cls(voxels, voxel_size)


def _eq_sides(sides, n):
    zeros = np.isclose(sides, 0)
    if all(zeros):
        # An empty or 1 point morphology should make only 1 (empty) voxel
        return np.array([1])
    elif any(zeros):
        # Remove any zeros, by fixing their dimensions to 1 (zero-width) partition
        solution = np.ones(len(sides))
        solution[~zeros] = _eq_sides(sides[~zeros], n)
        return solution
    elif len(sides) == 1:
        # Only 1 dimension, only 1 solution: all voxels along that dimension.
        return np.array([n])

    # Use the relative magnitudes of each side
    norm = sides / max(sides)
    # Find out how many divisions each side should to form a grid with `n` rhomboids.
    per_side = norm * (n / np.prod(norm)) ** (1 / len(sides))
    # Divisions should be integers, and minimum 1
    solution = np.maximum(np.floor(per_side), 1)
    order = np.argsort(sides)
    smallest = order[0]
    if len(sides) > 2:
        # Because of the integer rounding the product isn't necesarily optimal, so we keep
        # the safest (smallest) value, and solve the problem again in 1 less dimension.
        solved = solution[smallest]
        look_for = n / solved
        others = sides[order[1:]]
        solution[order[1:]] = _eq_sides(others, look_for)
    else:
        # In the final 2-dimensional case the remainder of the division is rounded off
        # to the nearest integer, giving the smallest error on the product and final
        # number of rhomboids in the grid.
        largest = order[1]
        solution[largest] = round(n / solution[smallest])
    return solution


# https://stackoverflow.com/a/24769712/1016004
def _is_broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def _squash_zero(arr):
    return np.where(np.isclose(arr, 0), np.finfo(float).max, arr)


__all__ = ["VoxelData", "VoxelSet"]
