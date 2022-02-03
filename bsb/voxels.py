from . import config
from .config import types
from .trees import BoxTree
import numpy as np
import functools
import abc
import nrrd


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
            self._data = np.array(data, copy=False)
            if self._data.ndim == 0:
                raise ValueError("Invalid non-sequence voxel data")
            elif self._data.ndim == 1:
                self._data = self._data.reshape(-1, 1)
            if len(self._data) != len(voxels):
                raise ValueError("`voxels` and `data` length unequal.")
        else:
            self._data = None
        if data_keys is None:
            data_keys = []
        self._data_keys = [*data_keys]

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

    def __len__(self):
        return len(self.get_raw(copy=False))

    def __getitem__(self, index):
        voxels = self.get_raw(copy=False)[index]
        if self._single_size:
            voxel_size = self._size
        else:
            voxel_size = self._sizes[index]
        if self.has_data:
            data = self._data[index]
            if data.ndim < 2:
                data.reshape(-1, self._data.shape[1])
        else:
            data = None
        if voxels.ndim == 0:
            raise Exception("holla")
        if voxels.ndim == 1:
            voxels = voxels.reshape(-1, 3)
        return VoxelSet(voxels, voxel_size, data)

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
        """
        return self._regular

    @property
    def of_equal_size(self):
        return self._single_size or len(np.unique(self._sizes, axis=0)) < 2

    @property
    def size(self):
        """
        The size of the voxels. When it is 0D or 1D it counts as the size for all voxels,
        if it is 2D it is 1 an individual size per voxel.

        :rtype: numpy.ndarray
        """
        return self.get_size()

    @property
    def data(self):
        """
        The size of the voxels. When it is 0D or 1D it counts as the size for all voxels,
        if it is 2D it is 1 an individual size per voxel.

        :rtype: Union[numpy.ndarray, None]
        """
        return self.get_data()

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
        voxels = cls(np.array([ldc]), np.array([mdc - ldc]))
        if data is not None:
            voxels._data = np.array([data], dtype=object)
        return voxels

    @classmethod
    def concatenate(cls, *sets):
        # Short circuit "stupid" concat requests
        if not sets:
            return cls.empty()
        elif len(sets) == 1:
            return sets[0].copy()

        if any(s.has_data for s in sets):
            data = np.concatenate([s.get_data(copy=False) for s in sets])
        else:
            data = None
        primer = None
        # Check which sets we are concatenating, maybe we can keep them in reduced data
        # forms. If they don't line up, we expand and concatenate the expanded forms.
        if all(
            # `primer` is assigned the first non-empty set, all sizes must match sizes can
            # still be 0D, 1D or 2D, but if they're allclose broadcasted it is fine! :)
            np.allclose(s.get_size(copy=False), primer.get_size(copy=False))
            for s in sets
            if (primer := primer or s)
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
        return VoxelSet(voxels, sizes, data=data, irregular=irregular)

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
                return np.array(self._data, copy=copy)
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

    def snap_to_grid(self, grid_size, unique=False):
        if self.regular:
            grid = self._indices // (grid_size / self._size)
        else:
            grid = self._coords // grid_size
        data = self._data
        if unique:
            if self.has_data:
                grid, id = np.unique(grid, return_index=True, axis=0)
                data = data[id]
            else:
                grid = np.unique(grid, axis=0)
        return VoxelSet(grid, grid_size, data)

    def resize(self, size):
        val = np.array(size, copy=False)
        if np.array(size).type.python_type is object:
            raise Exception("no no no")
        if val.ndim > 1:
            if val.size != len(self):
                raise Exception("Hell to the no")
            if self.regular:
                self._coords = self.as_spatial_coords()
                del self._indices
                self._regular = False
        self._size = size

    def select(self, ldc, mdc):
        data = self._data
        coords = self.as_spatial_coords(copy=False)
        inside = np.all(np.logical_and(ldc <= coords, coords < mdc), axis=1)
        return self[inside]

    def select_chunk(self, chunk):
        return self.select(chunk.ldc, chunk.mdc)

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

    def _boxes(self):
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
        return np.column_stack((ldc, mdc))

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
        branch_vcs = [
            b.as_matrix(with_radius=False) // voxel_size for b in morphology.branches
        ]
        if with_data:
            voxel_reduce = {}
            for branch, point_vcs in enumerate(branch_vcs):
                for point, point_vc in enumerate(point_vcs):
                    voxel_reduce.setdefault(tuple(point_vc), []).append((branch, point))
            voxels = np.array(tuple(voxel_reduce.keys()))
            data = np.array(list(voxel_reduce.values()), dtype=object)
            return cls(voxels, voxel_size, data=data)
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
    per_side = norm * (n / np.product(norm)) ** (1 / len(sides))
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


def _safe_zero_div(arr):
    return np.where(np.isclose(arr, 0), np.finfo(float).max, arr)
