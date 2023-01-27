"""
    Module for the Partition configuration nodes and its dependencies.
"""

from ._layout import Layout, RhomboidData
from .. import config
from ..config import types
from ..exceptions import (
    RequirementError,
    LayoutError,
    ConfigurationError,
    AllenApiError,
    NodeNotFoundError,
)
from ..storage._files import NrrdDependencyNode
from ..storage._util import _cached_file
from ..voxels import VoxelSet
from ..storage import Chunk
from ..reporting import report
import numpy as np
import collections
import functools
import requests
import nrrd
import json
import abc


def _size_requirements(section):
    if "thickness" not in section and "volume_scale" not in section:
        raise RequirementError(
            "Either a `thickness` or `volume_scale` attribute required"
        )


class _backref_property(property):
    def __backref__(self, instance, value):
        setattr(instance, "_region", value)


@config.dynamic(
    attr_name="type",
    required=False,
    default="layer",
    auto_classmap=True,
)
class Partition(abc.ABC):
    name = config.attr(key=True)

    @_backref_property
    def region(self):
        return self._region

    @property
    def data(self):
        # The data property is read-only to users, but `_data` is assigned
        # during the layout process
        return self._data

    @abc.abstractmethod
    def volume(self, chunk=None):  # pragma: nocover
        """
        Calculate the volume of the partition in μm^3.

        :param chunk: If given, limit the volume of the partition inside of the chunk.
        :type chunk: bsb.storage.Chunk
        :returns: Volume of the partition (in the chunk)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def surface(self, chunk=None):  # pragma: nocover
        """
        Calculate the surface of the partition in μm^2.

        :param chunk: If given, limit the surface of the partition inside of the chunk.
        :type chunk: bsb.storage.Chunk
        :returns: Surface of the partition (in the chunk)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def to_chunks(self, chunk_size):  # pragma: nocover
        """
        Calculate all the chunks this partition occupies when cut into ``chunk_sized``
        pieces.

        :param chunk_size: Size per chunk (in μm). The slicing always starts at [0, 0, 0].
        :type chunk_size: numpy.ndarray
        :returns: Chunks occupied by this partition
        :rtype: List[bsb.storage.Chunk]
        """
        pass

    @abc.abstractmethod
    def chunk_to_voxels(self, chunk):  # pragma: nocover
        """
        Voxelize the partition's occupation in this chunk. Required to fill the partition
        with cells by the placement module.

        :param chunk: The chunk to calculate voxels for.
        :type chunk: bsb.storage.Chunk
        :returns: The set of voxels that together make up the shape of this partition in
          this chunk.
        :rtype: bsb.voxels.VoxelSet
        """
        pass

    @abc.abstractmethod
    def rotate(self, rotation):  # pragma: nocover
        """
        Rotate the partition by the given rotation object.

        :param rotation: Rotation object.
        :type rotation: scipy.spatial.transform.Rotation
        :raises: :class:`.exceptions.LayoutError` if the rotation needs to be rejected.
        """
        pass

    @abc.abstractmethod
    def translate(self, offset):  # pragma: nocover
        """
        Translate the partition by the given offset.

        :param offset: Offset, XYZ.
        :type offset: numpy.ndarray
        :raises: :class:`.exceptions.LayoutError` if the translation needs to be rejected.
        """
        pass

    @abc.abstractmethod
    def scale(self, factors):  # pragma: nocover
        """
        Scale up/down the partition according to the given factors.

        :param factors: Scaling factors, XYZ.
        :type factors: numpy.ndarray
        :raises: :class:`.exceptions.LayoutError` if the scaling needs to be rejected.
        """
        pass

    @abc.abstractmethod
    def get_layout(self, hint):  # pragma: nocover
        """
        Given a Layout as hint to begin from, create a Layout object that describes how
        this partition would like to be laid out.

        :param hint: The layout space that this partition should place itself in.
        :type hint: bsb.topology._layout.Layout
        :returns: The layout describing the space this partition takes up.
        :rtype: bsb.topology._layout.Layout
        """
        pass


@config.node
class Rhomboid(Partition, classmap_entry="rhomboid"):
    dimensions = config.attr(type=types.list(type=float, size=3))
    can_scale = config.attr(type=bool, default=True)
    origin = config.attr(type=types.list(type=float, size=3))
    can_move = config.attr(type=bool, default=True)
    orientation = config.attr(type=types.list(type=float, size=3))
    can_rotate = config.attr(type=bool, default=True)

    def volume(self, chunk=None):
        if chunk is not None:
            # Create an intersection between the partition and the chunk
            low = np.maximum(self.ldc, chunk.ldc)
            high = np.minimum(self.mdc, chunk.mdc)
            return np.product(np.maximum(high - low, 0))
        else:
            return np.product(self.data.dimensions)

    @property
    def mdc(self):
        return self._data.mdc

    @property
    def ldc(self):
        return self._data.ldc

    def surface(self, chunk=None):
        if chunk is not None:
            # Gets the xz "square" from a volume
            sq = lambda v: np.array(v)[[0, 2]]
            ldc = sq(self.ldc)
            mdc = sq(self.mdc)
            cl = sq(chunk.ldc)
            cm = sq(chunk.mdc)
            # Create an intersection between the partition and the chunk
            low = np.maximum(ldc, cl)
            high = np.minimum(mdc, cm)
            return np.product(np.maximum(high - low, 0))
        else:
            return self.data.width * self.data.depth

    def to_chunks(self, chunk_size):
        # Get the low and high range of the data in chunk coordinates
        low_r = np.floor(self.ldc / chunk_size).astype(int)
        high_r = np.ceil(self.mdc / chunk_size).astype(int)
        # Create a grid that includes all the chunk coordinates within those data
        coords = np.mgrid[tuple(range(low, high) for low, high in zip(low_r, high_r))]
        # Order the coordinate grid into a list of chunk coordinates.
        return np.column_stack(tuple(dim.ravel() for dim in coords))

    def chunk_to_voxels(self, chunk):
        """
        Return an approximation of this partition intersected with a chunk as a list of
        voxels.

        Default implementation creates a parallellepepid intersection between the
        LDC, MDC and chunk data.
        """
        low = np.maximum(self.ldc, chunk.ldc)
        high = np.minimum(self.mdc, chunk.mdc)
        # Return 0 voxels when the coords are OOB for this partition
        if np.any(low > high):
            return VoxelSet.empty()
        else:
            return VoxelSet.one(low, high)

    def rotate(self, rot):
        raise LayoutError("Rotation not implemented yet.")

    def translate(self, translation):
        self.data.ldc += translation
        self.data.mdc += translation

    def scale(self, factors):
        self.data.mdc = self.data.ldc + (self.data.mdc - self.data.ldc) * factors

    def get_dependencies(self):
        """
        Return other partitions or regions that need to be laid out before this.
        """
        return []

    def get_layout(self, hint):
        if self.dimensions is None:
            dim = hint.data.mdc - hint.data.ldc
        else:
            dim = self.dimensions
        if self.origin is None:
            orig = hint.data.ldc.copy()
        else:
            orig = self.origin
        return Layout(RhomboidData(orig, dim + orig), owner=self)


@config.node
class Layer(Rhomboid, classmap_entry="layer"):
    dimensions = config.unset()
    thickness = config.attr(type=float, required=_size_requirements)
    volume_scale = config.attr(
        type=types.or_(
            types.list(float, size=2),
            types.scalar_expand(
                float,
                lambda x: [x, x],
            ),
        ),
        default=lambda: [1.0, 1.0],
        call_default=True,
    )
    axis = config.attr(type=types.in_(["x", "y", "z"]), default="y")
    stack_index = config.attr(type=float, default=0)

    def get_layout(self, hint):
        axis = ["x", "y", "z"].index(self.axis)
        dim = hint.data.mdc - hint.data.ldc
        dim[axis] = self.thickness
        if self.origin is None:
            orig = hint.data.ldc.copy()
        else:
            orig = self.origin
        return Layout(RhomboidData(orig, dim + orig), owner=self)

    # TODO: Layer scaling


@config.node
class Voxels(Partition, abc.ABC, classmap_entry=None):
    @abc.abstractmethod
    def get_voxelset(self):
        pass

    @functools.cached_property
    def voxelset(self):
        return self.get_voxelset()

    def to_chunks(self, chunk_size):
        return self.voxelset.snap_to_grid(chunk_size, unique=True)

    def _lookup(self, chunk):
        if not hasattr(self, "_map"):
            vs = self.voxelset.snap_to_grid(chunk.dimensions)
            map = {}
            for i, idx in enumerate(vs):
                map.setdefault(idx.view(Chunk), []).append(i)
            self._map = {k: self.voxelset[v] for k, v in map.items()}
        return self._map

    def chunk_to_voxels(self, chunk):
        return self._lookup(chunk).get(chunk, VoxelSet.empty())

    def get_layout(self, hint):
        return Layout(RhomboidData(*self.voxelset.bounds), owner=self)

    def rotate(self, rotation):
        raise LayoutError("Axis-aligned voxelsets can't be rotated.")

    def scale(self, factor):
        raise LayoutError("Voxelset scaling not supported.")

    def translate(self, offset):
        raise LayoutError("Voxelset translation not supported.")

    def surface(self, chunk=None):
        raise LayoutError("Voxelset surface calculations not supported.")

    def volume(self, chunk=None):
        if chunk is not None:
            vs = self.chunk_to_voxels(chunk)
        else:
            vs = self.voxelset
        return vs.volume


@config.node
class NrrdVoxels(Voxels, classmap_entry="nrrd"):
    source = config.attr(
        type=NrrdDependencyNode,
        required=types.mut_excl("source", "sources", required=True),
    )
    sources = config.attr(
        type=types.list(NrrdDependencyNode),
        required=types.mut_excl("source", "sources", required=True),
    )
    mask_value = config.attr(type=int)
    mask_source = config.attr(type=NrrdDependencyNode)
    mask_only = config.attr(type=bool, default=False)
    voxel_size = config.attr(type=types.voxel_size(), required=True)
    keys = config.attr(type=types.list(str))
    sparse = config.attr(type=bool, default=True)
    strict = config.attr(type=bool, default=True)

    def get_mask(self):
        mask_shape = self._validate()
        mask = np.zeros(mask_shape, dtype=bool)
        if self.sparse:
            # Use integer (sparse) indexing
            mask = [np.empty((0,), dtype=int) for i in range(3)]
            for mask_src in self._mask_src:
                mask_data = mask_src.get_data()
                new_mask = np.nonzero(self._mask_cond(mask_data))
                for i, mask_vector in enumerate(new_mask):
                    mask[i] = np.concatenate((mask[i], mask_vector))
            inter = np.unique(mask, axis=1)
            mask = tuple(inter[i, :] for i in range(3))
        else:
            # Use boolean (dense) indexing
            for mask_src in self._mask_src:
                mask_data = mask_src.get_data()
                mask = mask | self._mask_cond(mask_data)
            mask = np.nonzero(mask)
        return mask

    def get_voxelset(self):
        mask = self.get_mask()
        voxel_data = None
        if not self.mask_only:
            voxel_data = np.empty((len(mask[0]), len(self._src)))
            for i, source in enumerate(self._src):
                voxel_data[:, i] = source.get_data()[mask]

        return VoxelSet(
            np.transpose(mask),
            self.voxel_size,
            data=voxel_data,
            data_keys=self.keys,
        )

    def _validate(self):
        self._validate_sources()
        shape = self._validate_source_compat()
        self._validate_mask_condition()
        return shape

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
        mask_headers = {s: s.get_header() for s in self._mask_src}
        source_headers = {s: s.get_header() for s in self._src}
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
class AllenStructure(NrrdVoxels, classmap_entry="allen"):
    struct_id = config.attr(
        type=int, required=types.mut_excl("struct_id", "struct_name", required=True)
    )
    struct_name = config.attr(
        type=types.str(strip=True, lower=True),
        required=types.mut_excl("struct_id", "struct_name", required=True),
    )
    source = config.attr(
        type=NrrdDependencyNode,
        required=types.mut_excl("source", "sources", required=False),
    )
    sources = config.attr(
        type=types.list(NrrdDependencyNode),
        required=types.mut_excl("source", "sources", required=False),
        default=list,
        call_default=True,
    )

    @config.property
    def voxel_size(self):
        return 25

    @config.property
    def mask_only(self):
        return self.source is None and len(self.sources) == 0

    @config.property
    @functools.cache
    def mask_source(self):
        node = NrrdDependencyNode()
        node._file = _cached_file(
            "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd",
        )
        return node

    @classmethod
    @functools.cache
    def _dl_structure_ontology(cls):
        return json.loads(
            _cached_file(
                "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
            ).get_content()[0]
        )["msg"]

    @classmethod
    def get_structure_mask_condition(cls, find):
        """
        Return a lambda that when applied to the mask data, returns a mask that delineates
        the Allen structure.

        :param find: Acronym or ID of the Allen structure.
        :type find: Union[str, int]
        :returns: Masking lambda
        :rtype: Callable[numpy.ndarray]
        """
        mask = cls.get_structure_idset(find)
        if len(mask) > 1:
            return lambda data: np.isin(data, mask)
        else:
            mask0 = mask[0]
            return lambda data: data == mask0

    @classmethod
    def get_structure_mask(cls, find):
        mask_data, _ = nrrd.read(cls._dl_mask())
        return cls.get_structure_mask_condition(find)(mask_data)

    @classmethod
    def get_structure_idset(cls, find):
        """
        Return the set of IDs that make up the requested Allen structure.

        :param find: Acronym or ID of the Allen structure.
        :type find: Union[str, int]
        :returns: Set of IDs
        :rtype: numpy.ndarray
        """
        struct = cls.find_structure(find)
        values = set()

        def flatmask(item):
            values.add(item["id"])

        cls._visit_structure([struct], flatmask)
        return np.array([*values], dtype=int)

    @classmethod
    def find_structure(cls, id):
        """
        Find an Allen structure by name, acronym or ID.

        :param id: Query for the name, acronym or ID of the Allen structure.
        :type id: Union[str, int, float]
        :returns: Allen structure node of the Allen ontology tree.
        :rtype: dict
        :raises: NodeNotFoundError
        """
        if isinstance(id, str):
            treat = lambda s: s.strip().lower()
            name = treat(id)
            find = lambda x: treat(x["name"]) == name or treat(x["acronym"]) == name
        elif isinstance(id, int) or isinstance(id, float):
            id = int(id)
            find = lambda x: x["id"] == id
        else:
            raise TypeError(f"Argument must be a string or a number. {type(id)} given.")
        try:
            return cls._find_structure(find)
        except NodeNotFoundError:
            raise NodeNotFoundError(f"Could not find structure '{id}'") from None

    @classmethod
    def _find_structure(cls, find):
        result = None

        def visitor(item):
            nonlocal result
            if find(item):
                result = item
                return True

        tree = cls._dl_structure_ontology()
        cls._visit_structure(tree, visitor)
        if result is None:
            raise NodeNotFoundError("Could not find a node that satisfies constraints.")
        return result

    @classmethod
    def _visit_structure(cls, tree, visitor):
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
        # We override the `NrrdVoxels`' `_validate_mask_condition` and use this
        # function as a hook to find and set the mask condition to select every voxel that
        # has an id that is part of the structure.
        id = self.struct_id if self.struct_id is not None else self.struct_name
        self._mask_cond = self.get_structure_mask_condition(id)


def _safe_hread(s):
    try:
        return nrrd.read_header(s)
    except StopIteration:
        raise IOError(f"Empty NRRD file '{s}' could not be read.") from None


def _repeat_first():
    _set = False
    first = None

    def repeater(val):
        nonlocal _set, first
        if not _set:
            first, _set = val, True
        return first

    return repeater
