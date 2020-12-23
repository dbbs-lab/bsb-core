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

    @property
    def volume(self):
        return np.product(self.boundaries.dimensions)

    def to_chunks(*args):
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])


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
