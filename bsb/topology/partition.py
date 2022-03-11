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


@config.dynamic(
    attr_name="type",
    type=types.in_classmap(),
    required=False,
    default="layer",
    auto_classmap=True,
)
class Partition(abc.ABC):
    name = config.attr(key=True)
    region = config.ref(region_ref, populate="children")

    @abc.abstractmethod
    def volume(self, chunk=None):
        pass

    @abc.abstractmethod
    def surface(self, chunk=None):
        pass

    @abc.abstractmethod
    def to_chunks(self, chunk_size):
        pass

    @abc.abstractmethod
    def chunk_to_voxels(self, chunk):
        pass

    @abc.abstractmethod
    def rotate(self, rotation):
        pass

    @abc.abstractmethod
    def translate(self, offset):
        pass

    @abc.abstractmethod
    def scale(self, factors):
        pass

    @abc.abstractmethod
    def get_layout(self, hint):
        pass


@config.node
class Rhomboid(Partition, classmap_entry="rhomboid"):
    dataclass = RhomboidData
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
        return self.data.mdc

    @property
    def ldc(self):
        return self.data.ldc

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
            orig = hint.data.ldc
        else:
            orig = self.origin
        return Layout(RhomboidData(orig, dim - orig), owner=self)


@config.node
class Layer(Rhomboid, classmap_entry="layer"):
    thickness = config.attr(type=float, required=_size_requirements)
    xz_scale = config.attr(
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
    xz_center = config.attr(type=bool, default=False)
    stack_index = config.attr(type=float, default=0)

    # TODO: Layer stacking
    # TODO: Layer scaling
    # TODO: Layer centering


@config.node
class Voxels(Partition, classmap_entry="voxels"):
    voxels = config.attr(type=VoxelLoader, required=True)

    @functools.cached_property
    def voxelset(self):
        return self.voxels.get_voxelset()

    def to_chunks(self, chunk_size):
        print(
            "Thalamus as",
            len(self.voxelset.snap_to_grid(chunk_size, unique=True)),
            "chunks",
        )
        return self.voxelset.snap_to_grid(chunk_size, unique=True)

    def chunk_to_voxels(self, chunk):
        if not hasattr(self, "_map"):
            vs = self.voxelset.snap_to_grid(chunk.dimensions)
            map = {}
            for i, idx in enumerate(vs):
                map.setdefault(idx.view(Chunk).id, []).append(i)
            self._map = {k: self.voxelset[v] for k, v in map.items()}
        return self._map.get(chunk, VoxelSet.empty())

    def layout(self, boundaries):
        # Buondaries are currently the network dimensions in JSON file
        self.boundaries = boundaries
