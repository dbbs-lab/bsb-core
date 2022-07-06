"""
    Module for the Partition configuration nodes and its dependencies.
"""

import functools
import abc
from ._layout import Layout, RhomboidData
from .. import config
from ..config import types
from ..config.refs import region_ref
from ..exceptions import *
from ..voxels import VoxelSet, VoxelLoader
from ..storage import Chunk
import numpy as np


def _size_requirements(section):
    if "thickness" not in section and "volume_scale" not in section:
        raise RequirementError(
            "Either a `thickness` or `volume_scale` attribute required"
        )


class _prop(property):
    pass


@config.dynamic(
    attr_name="type",
    type=types.in_classmap(),
    required=False,
    default="layer",
    auto_classmap=True,
)
class Partition(abc.ABC):
    name = config.attr(key=True)

    @_prop
    def region(self):
        return self._region

    region.__backref__ = lambda self, value: setattr(self, "_region", value)

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
class Voxels(Partition, classmap_entry="voxels"):
    voxels = config.attr(type=VoxelLoader, required=True)

    @functools.cached_property
    def voxelset(self):
        return self.voxels.get_voxelset()

    def to_chunks(self, chunk_size):
        return self.voxelset.snap_to_grid(chunk_size, unique=True)

    def _lookup(self, chunk):
        if not hasattr(self, "_map"):
            vs = self.voxelset.snap_to_grid(chunk.dimensions)
            map = {}
            for i, idx in enumerate(vs):
                map.setdefault(idx.view(Chunk).id, []).append(i)
            self._map = {k: self.voxelset[v] for k, v in map.items()}
        return self._map

    def chunk_to_voxels(self, chunk):
        return self._lookup(chunk).get(chunk, VoxelSet.empty())

    def get_layout(self, hint):
        return Layout(RhomboidData(*self.voxelset.bounds), owner=self)

    def rotate(self):
        raise LayoutError("Axis-aligned voxelsets can't be rotated.")

    def scale(self):
        raise LayoutError("Voxelset scaling not supported.")

    def surface(self):
        raise LayoutError("Voxelset surface calculations not supported.")

    def translate(self):
        raise LayoutError("Voxelset translation not supported.")

    def volume(self, chunk=None):
        if chunk is not None:
            vs = self.chunk_to_voxels(chunk)
        else:
            vs = self.voxelset
        return vs.volume
