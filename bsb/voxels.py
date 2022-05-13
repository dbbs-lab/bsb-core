from . import config
from .config import types
from .trees import BoxTree
from .exceptions import *
from .reporting import report
import numpy as np
import json
import functools
import itertools
import collections
import abc
import requests
import nrrd


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

    def __len__(self):
        return len(self.get_raw(copy=False))

    def __getitem__(self, index):
        if self.has_data:
            data = self._data[index]
            index, _, _ = self._data._split_index(index)
        else:
            data, keys = None, None
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
                    insert += f"with {self._data.shape[1]} data columns "
            else:
                insert += "without data "
        return obj.replace("at 0x", insert + "at 0x")

    def __repr__(self):
        return self.__str__()

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
            for l, fill in zip(ln, fillers):
                if fill is not None:
                    if not fill.keys:
                        cols = slice(None, fill.shape[1])
                    else:
                        cols = [keys.index(key) for key in fill.keys]
                    data[ptr : (ptr + l), cols] = fill
                    ptr += l
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

    def snap_to_grid(self, grid_size, unique=False):
        if self.regular:
            grid = self._indices // _squash_zero(grid_size / _squash_zero(self._size))
        else:
            grid = self._coords // _squash_zero(grid_size)
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
        data = self._data
        coords = self.as_spatial_coords(copy=False)
        inside = np.all(np.logical_and(ldc <= coords, coords < mdc), axis=1)
        return self[inside]

    def crop_chunk(self, chunk):
        return self.crop(chunk.ldc, chunk.mdc)

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
        branch_vcs = [b.points // _squash_zero(voxel_size) for b in morphology.branches]
        if with_data:
            voxel_reduce = {}
            for branch, point_vcs in enumerate(branch_vcs):
                for point, point_vc in enumerate(point_vcs):
                    voxel_reduce.setdefault(tuple(point_vc), []).append((branch, point))
            voxels = np.array(tuple(voxel_reduce.keys()))
            data = np.array(list(voxel_reduce.values()), dtype=object)
            return cls(voxels, voxel_size, data=data)
        else:
            voxels = np.unique(np.concatenate(branch_vcs), axis=0)
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
    source = config.attr(
        type=str, required=types.mut_excl("source", "sources", required=True)
    )
    sources = config.attr(
        type=types.list(str), required=types.mut_excl("source", "sources", required=True)
    )
    mask_value = config.attr(type=int)
    mask_source = config.attr(type=str)
    mask_only = config.attr(type=bool, default=False)
    voxel_size = config.attr(type=types.voxel_size(), required=True)
    keys = config.attr(type=types.list(str))
    sparse = config.attr(type=bool, default=True)
    strict = config.attr(type=bool, default=True)

    def get_voxelset(self):
        mask_shape = self._validate()
        mask = np.zeros(mask_shape, dtype=bool)
        if self.sparse:
            # Use integer (sparse) indexing
            mask = [np.empty((0,), dtype=int) for i in range(3)]
            for mask_src in self._mask_src:
                mask_data, _ = nrrd.read(mask_src)
                new_mask = np.nonzero(self._mask_cond(mask_data))
                for i, mask_vector in enumerate(new_mask):
                    mask[i] = np.concatenate((mask[i], mask_vector))
            inter = np.unique(mask, axis=1)
            mask = tuple(inter[i, :] for i in range(3))
        else:
            # Use boolean (dense) indexing
            for mask_src in self._mask_src:
                mask_data, _ = nrrd.read(mask_src)
                mask = mask | self._mask_cond(mask_data)
            mask = np.nonzero(mask)

        if not self.mask_only:
            voxel_data = np.empty((len(mask[0]), len(self._src)))
            for i, source in enumerate(self._src):
                data, _ = nrrd.read(source)
                voxel_data[:, i] = data[mask]

        return VoxelSet(
            np.transpose(mask),
            self.voxel_size,
            data=voxel_data if not self.mask_only else None,
            data_keys=self.keys,
        )

    def _validate(self):
        self._validate_sources()
        self._validate_source_compat()
        self._validate_mask_condition()

    def _validate_sources(self):
        if self.source is not None:
            self._src = [self.source]
        else:
            self._src = self.sources.copy()
        if self.mask_source is not None:
            self._mask_src = [self.mask_source]
        else:
            self._mask_src = self._src.copy()

    def _validate_source_compat(self):
        mask_headers = {s: nrrd.read_header(s) for s in self._mask_src}
        source_headers = {s: nrrd.read_header(s) for s in self._src}
        all_headers = mask_headers.copy()
        all_headers.update(source_headers)
        dim_probs = [(s, d) for s, h in all_headers.items() if (d := h["dimension"]) != 3]
        if dim_probs:
            summ = ", ".join(f"'{s}' has {d}" for s, d in dim_probs)
            raise ConfigurationError(f"NRRD voxels must contain 3D arrays; {summ}")
        mask_sizes = {s: [*h["sizes"]] for s, h in mask_headers.items()}
        source_sizes = {s: [*h["sizes"]] for s, h in source_headers.items()}
        all_sizes = mask_sizes.copy()
        all_sizes.update(source_sizes)
        mask_shape = np.maximum.reduce([*mask_sizes.values()])
        if self.mask_only:
            src_shape = mask_shape
        else:
            src_shape = np.minimum.reduce([*source_sizes.values()])
        first = _repeat_first()
        # Check for any size mismatch
        if self.strict and any(size != first(size) for size in all_sizes.values()):
            raise ConfigurationError(
                f"NRRD file size mismatch in `{self.get_node_name()}`: {all_sizes}"
            )
        elif np.any(mask_shape > src_shape):
            raise ConfigurationError(
                f"NRRD mask too big; it may select OOB source voxels:"
                + f" {mask_shape} > {src_shape}"
            )
        return mask_shape

    def _validate_mask_condition(self):
        if self.mask_value:
            self._mask_cond = lambda data: data == self.mask_value
        else:
            self._mask_cond = lambda data: data != 0


@config.node
class AllenStructureLoader(NrrdVoxelLoader, classmap_entry="allen"):
    struct_id = config.attr(
        type=int, required=types.mut_excl("struct_id", "struct_name", required=True)
    )
    struct_name = config.attr(
        type=types.str(strip=True, lower=True),
        required=types.mut_excl("struct_id", "struct_name", required=True),
    )
    source = config.attr(
        type=str, required=types.mut_excl("source", "sources", required=False)
    )
    sources = config.attr(
        type=types.list(str),
        required=types.mut_excl("source", "sources", required=False),
        default=list,
        call_default=True,
    )

    @config.property
    def mask_only(self):
        return self.source is None and len(self.sources) == 0

    @config.property
    @functools.cache
    def mask_source(self):
        from .storage import _util as _storutil

        url = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd"
        fname = "_annotations_25.nrrd.cache"
        link = _storutil.cachelink(fname, binary=True)
        if link.should_update():
            with link.set() as f:
                report("Downloading Allen Brain Atlas annotations", level=3)
                content = requests.get(url).content
                f.write(requests.get(url).content)
        else:
            report("Using cached Allen Brain Atlas annotations", level=4)
        return str(link.path)

    @functools.cache
    def _dl_structure_ontology(self):
        from .storage import _util as _storutil

        url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
        fname = "_allen_ontology.cache"
        link = _storutil.cachelink(fname)
        if link.should_update():
            report("Downloading Allen Brain Atlas structure ontology", level=3)
            payload = requests.get(url).json()
            if not payload.get("success", False):
                raise AllenApiError(f"Could not fetch ontology from Allen API at '{url}'")
            with link.set() as f:
                json.dump(payload["msg"], f)
        else:
            report("Using cached Allen Brain Atlas ontology", level=4)
        with link.get() as f:
            return json.load(f)

    def get_structure_mask_condition(self, find):
        mask = self.get_structure_mask(find)
        if len(mask) > 1:
            return lambda data: np.isin(data, mask)
        else:
            mask0 = mask[0]
            return lambda data: data == mask0

    def get_structure_mask(self, find):
        struct = self.find_structure(find)
        values = set()

        def flatmask(item):
            values.add(item["id"])

        self._visit_structure([struct], flatmask)
        return np.array([*values], dtype=int)

    @functools.singledispatchmethod
    def find_structure(self, id):
        find = lambda x: x["id"] == id
        try:
            return self._find_structure(find)
        except NodeNotFoundError:
            raise NodeNotFoundError(f"Could not find structure with id '{id}'") from None

    @find_structure.register
    def _(self, name: str):
        proc = lambda s: s.strip().lower()
        _name = proc(name)
        find = lambda x: proc(x["name"]) == _name or proc(x["acronym"]) == _name
        try:
            return self._find_structure(find)
        except NodeNotFoundError:
            raise NodeNotFoundError(
                f"Could not find structure with name '{name}'"
            ) from None

    def _find_structure(self, find):
        result = None

        def visitor(item):
            nonlocal result
            if find(item):
                result = item
                return True

        tree = self._dl_structure_ontology()
        self._visit_structure(tree, visitor)
        if result is None:
            raise NodeNotFoundError("Could not find a node that satisfies constraints.")
        return result

    def _visit_structure(self, tree, visitor):
        deck = collections.deque(tree)
        while True:
            try:
                item = deck.popleft()
            except IndexError:
                break
            if visitor(item):
                break
            deck.extend(item["children"])

    def _validate_mask_condition(self):
        id = self.struct_id if self.struct_id is not None else self.struct_name
        self._mask_cond = self.get_structure_mask_condition(id)


def _repeat_first():
    _set = False
    first = None

    def repeater(val):
        nonlocal _set, first
        if not _set:
            first, _set = val, True
        return first

    return repeater


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


def _squash_zero(arr):
    return np.where(np.isclose(arr, 0), np.finfo(float).max, arr)
