"""
    Module for the Partition configuration nodes and its dependencies.
"""

import abc
import collections
import functools
import json
import typing

import nrrd
import numpy as np

from .. import config
from ..config import types
from ..exceptions import (
    AllenApiError,
    ConfigurationError,
    LayoutError,
    NodeNotFoundError,
)
from ..storage._chunks import Chunk
from ..storage._files import NrrdDependencyNode
from ..storage._util import _cached_file
from ..voxels import VoxelSet
from ._layout import Layout, RhomboidData

if typing.TYPE_CHECKING:
    from ..core import Scaffold


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
    """
    Abstract class to describe spatial containers for network pieces.
    """

    scaffold: "Scaffold"
    name: str = config.attr(key=True)

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
        :type chunk: bsb.storage._chunks.Chunk
        :returns: Volume of the partition (in the chunk)
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def surface(self, chunk=None):  # pragma: nocover
        """
        Calculate the surface of the partition in μm^2.

        :param chunk: If given, limit the surface of the partition inside of the chunk.
        :type chunk: bsb.storage._chunks.Chunk
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
        :rtype: List[bsb.storage._chunks.Chunk]
        """
        pass

    @abc.abstractmethod
    def chunk_to_voxels(self, chunk):  # pragma: nocover
        """
        Voxelize the partition's occupation in this chunk. Required to fill the partition
        with cells by the placement module.

        :param chunk: The chunk to calculate voxels for.
        :type chunk: bsb.storage._chunks.Chunk
        :returns: The set of voxels that together make up the shape of this partition in
          this chunk.
        :rtype: bsb.voxels.VoxelSet
        """
        pass

    def to_voxels(self):
        """
        Voxelize the partition's occupation.
        """

        chunk_size = self.scaffold.network.chunk_size
        return VoxelSet.concatenate(
            *(
                self.chunk_to_voxels(Chunk(chunk, chunk_size))
                for chunk in self.to_chunks(chunk_size)
            )
        )

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
    """
    Rectangular cuboid partition defined according to its origin and dimensions.
    """

    dimensions: list[float] = config.attr(type=types.list(type=float, size=3))
    """Sizes of the partition for each axis."""
    can_scale: bool = config.attr(type=bool, default=True)
    """Boolean flag to authorize rescaling of the partition dimensions"""
    origin: list[float] = config.attr(type=types.list(type=float, size=3))
    """Coordinate of the origin of the partition"""
    can_move: bool = config.attr(type=bool, default=True)
    """Boolean flag to authorize the translation of the partition"""
    orientation: list[float] = config.attr(type=types.list(type=float, size=3))
    can_rotate: bool = config.attr(type=bool, default=True)
    """Boolean flag to authorize the rotation of the partition"""

    def volume(self, chunk=None):
        if chunk is not None:
            # Create an intersection between the partition and the chunk
            low = np.maximum(self.ldc, chunk.ldc)
            high = np.minimum(self.mdc, chunk.mdc)
            return np.prod(np.maximum(high - low, 0))
        else:
            return np.prod(self.data.dimensions)

    @property
    def mdc(self):
        """
        Return the highest coordinate of the partition.
        """
        return self._data.mdc

    @property
    def ldc(self):
        """
        Return the lowest coordinate of the partition.
        """
        return self._data.ldc

    def surface(self, chunk=None):
        if chunk is not None:
            # Gets the xz "square" from a volume
            sq = lambda v: np.array(v)[[0, 1]]
            ldc = sq(self.ldc)
            mdc = sq(self.mdc)
            cl = sq(chunk.ldc)
            cm = sq(chunk.mdc)
            # Create an intersection between the partition and the chunk
            low = np.maximum(ldc, cl)
            high = np.minimum(mdc, cm)
            return np.prod(np.maximum(high - low, 0))
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

        Default implementation creates a parallelepiped intersection between the
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
            dim = np.array(self.dimensions)
        if self.origin is None:
            orig = hint.data.ldc.copy()
        else:
            orig = np.array(self.origin)
        return Layout(RhomboidData(orig, dim + orig), owner=self)


@config.node
class Layer(Rhomboid, classmap_entry="layer"):
    """
    Partition that occupies the full space of its containing region
    except on a defined axis, where it is limited. This creates a stratum
    within the region along the chosen axis.
    """

    dimensions = config.unset()
    thickness: float = config.attr(type=float, required=True)
    """Thickness of the layer along its axis"""
    axis: typing.Union[typing.Literal["x"], typing.Literal["y"], typing.Literal["z"]] = (
        config.attr(type=types.in_(["x", "y", "z"]), default="z")
    )
    """Axis along which the layer will be limited."""

    def get_layout(self, hint):
        axis = ["x", "y", "z"].index(self.axis)
        dim = hint.data.mdc - hint.data.ldc
        dim[axis] = self.thickness
        if self.origin is None:
            orig = hint.data.ldc.copy()
        else:
            orig = np.array(self.origin)
        return Layout(RhomboidData(orig, dim + orig), owner=self)

    # TODO: Layer scaling


@config.node
class Voxels(Partition, abc.ABC, classmap_entry=None):
    """
    Partition based on a set of voxels.
    """

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
    """
    Voxel partition whose voxelset is loaded from an NRRD file. By default it includes all the
    nonzero voxels in the file, but other masking conditions can be specified. Additionally, data
    can be associated to each voxel by inclusion of (multiple) source NRRD files.
    """

    source: NrrdDependencyNode = config.attr(
        type=NrrdDependencyNode,
        required=types.mut_excl("source", "sources", required=False),
    )
    """Path to the NRRD file containing volumetric data to associate with the partition.
    If source is set, then sources should not be set."""
    sources: NrrdDependencyNode = config.list(
        type=NrrdDependencyNode,
        required=types.mut_excl("source", "sources", required=False),
    )
    """List of paths to NRRD files containing volumetric data to associate with the Partition.
    If sources is set, then source should not be set."""
    mask_value: int = config.attr(type=int)
    """Integer value to filter in mask_source (if it is set, otherwise sources/source) to create a 
    mask of the voxel set(s) used as input."""
    mask_source: NrrdDependencyNode = config.attr(type=NrrdDependencyNode)
    """Path to the NRRD file containing the volumetric annotation data of the Partition."""
    mask_only: bool = config.attr(type=bool, default=False)
    """Flag to indicate no voxel data needs to be stored"""
    voxel_size: int = config.attr(type=int, required=True)
    """Size of each voxel."""
    keys: list[str] = config.attr(type=types.list(str))
    """List of names to assign to each source of the Partition."""
    sparse: bool = config.attr(type=bool, default=True)
    """
    Boolean flag to expect a sparse or dense mask. If the mask selects most
    voxels, use ``dense``, otherwise use ``sparse``.
    """
    strict: bool = config.attr(type=bool, default=True)
    """Boolean flag to check the sources and the mask sizes. 
    When the flag is True, sources and mask should have exactly the same sizes;
    otherwise, sources sizes should be greater than mask sizes."""

    def get_mask(self):
        """
        Get the mask to apply on the sources' data of the partition.

        :returns: A tuple of arrays, one for each dimension of the mask, containing the indices of
            the non-zero elements in that dimension.
        """

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
        """
        Creates a VoxelSet of the sources of the Partition that matches its mask.

        :returns: VoxelSet of the Partition sources.
        """

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
    """
    Partition based on the Allen Institute for Brain Science mouse brain region ontology, later
    referred as Allen Mouse Brain Region Hierarchy (AMBRH)
    """

    struct_id: int = config.attr(
        type=int, required=types.mut_excl("struct_id", "struct_name", required=False)
    )
    """Id of the region to filter within the annotation volume according to the AMBRH.
    If struct_id is set, then struct_name should not be set."""
    struct_name: str = config.attr(
        type=types.str(strip=True, lower=True),
        required=types.mut_excl("struct_id", "struct_name", required=False),
        key=True,
    )
    """Name or acronym of the region to filter within the annotation volume according to the AMBRH.
    If struct_name is set, then struct_id should not be set."""

    @config.property(type=int)
    def voxel_size(self):
        """Size of each voxel."""
        return self._voxel_size if self._voxel_size is not None else 25

    @voxel_size.setter
    def voxel_size(self, value):
        self._voxel_size = value

    @config.property(type=bool)
    def mask_only(self):
        return self.source is None and len(self.sources) == 0

    @config.property(type=str)
    @functools.cache
    def mask_source(self):
        if hasattr(self, "_annotations_file"):
            return self._annotations_file
        else:
            node = NrrdDependencyNode()
            node._file = _cached_file(
                "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd",
            )
            return node

    @mask_source.setter
    def mask_source(self, value):
        self._mask_source = value
        if value is not None:
            self._annotations_file = NrrdDependencyNode(file=value)
        elif hasattr(self, "_annotations_file"):
            delattr(self, "_annotations_file")

    @classmethod
    @functools.cache
    def _dl_structure_ontology(cls):
        content = _cached_file(
            "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
        ).get_content()[0]
        try:
            return json.loads(content)["msg"]
        except json.decoder.JSONDecodeError:
            raise AllenApiError(
                "Could not parse the Allen mouse brain region hierarchy, "
                "most likely because the Allen API website is down. \n"
                "Here is the content retrieved: \n"
                f"{content}"
            )

    @classmethod
    def get_structure_mask_condition(cls, find):
        """
        Return a lambda that when applied to the mask data, returns a mask that delineates
        the Allen structure.

        :param find: Acronym, Name or ID of the Allen structure.
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
        """
        Returns the mask data delineated by the Allen structure.

        :param find: Acronym, Name or ID of the Allen structure.
        :type find: Union[str, int]
        :returns: A boolean of the mask filtered based on the Allen structure.
        :rtype: Callable[numpy.ndarray]
        """
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


__all__ = ["AllenStructure", "Layer", "NrrdVoxels", "Partition", "Rhomboid", "Voxels"]
