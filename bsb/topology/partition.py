"""
    Module for the Partition configuration nodes and its dependencies.
"""

from .. import config
from ..config import types
from ..config.refs import region_ref, partition_ref
from ..exceptions import *
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
class Partition:
    name = config.attr(key=True)
    region = config.ref(region_ref, populate="partitions", required=True)

    def layout(self, boundaries):
        return NotImplementedError("Partitions should define a `layout` method")

    def volume(self, chunk=None, chunk_size=None):
        if chunk is not None:
            # Create an intersection between the partition and the chunk
            low = np.maximum(self.boundaries.ldc, chunk * chunk_size)
            high = np.minimum(self.boundaries.mdc, (chunk + 1) * chunk_size)
            return np.product(np.maximum(high - low, 0))
        else:
            return np.product(self.boundaries.dimensions)

    def surface(self, chunk=None, chunk_size=None):
        if chunk is not None:
            # Gets the xz "square" from a volume
            sq = lambda v: np.array(v)[[0, 2]]
            ldc = sq(self.boundaries.ldc)
            mdc = sq(self.boundaries.mdc)
            chunk = sq(chunk)
            chunk_size = sq(chunk_size)
            # Create an intersection between the partition and the chunk
            low = np.maximum(ldc, chunk * chunk_size)
            high = np.minimum(mdc, (chunk + 1) * chunk_size)
            return np.product(np.maximum(high - low, 0))
        else:
            return self.boundaries.width * self.boundaries.depth

    def to_chunks(self, chunk_size):
        # Get the low and high range of the boundaries in chunk coordinates
        low_r = np.floor(self.boundaries.ldc / chunk_size).astype(int)
        high_r = np.ceil(self.boundaries.mdc / chunk_size).astype(int)
        # Create a grid that includes all the chunk coordinates within those boundaries
        coords = np.mgrid[tuple(range(low, high) for low, high in zip(low_r, high_r))]
        # Order the coordinate grid into a list of chunk coordinates.
        return np.column_stack(tuple(dim.ravel() for dim in coords))

    def chunk_to_voxels(self, chunk, chunk_size):
        """
        Return an approximation of this partition intersected with a chunk as a list of
        voxels.

        Default implementation creates a parallellepepid intersection between the
        LDC, MDC and chunk boundaries.
        """
        low = np.maximum(self.boundaries.ldc, chunk * chunk_size)
        high = np.minimum(self.boundaries.mdc, (chunk + 1) * chunk_size)
        return [[low, high - low]]


@config.node
class Layer(Partition, classmap_entry="layer"):
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
    z_index = config.attr(type=float, default=0)

    def get_dependencies(self):
        """
        Return other partitions or regions that need to be laid out before this.
        """
        return []

    def layout(self, boundaries):
        self.boundaries = boundaries
        boundaries.height = self.thickness

    # TODO: Layer stacking
    # TODO: Layer scaling
    # TODO: Layer centering
