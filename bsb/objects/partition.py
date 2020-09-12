"""
    Module for the Layer configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..config.refs import region_ref, layer_ref
from ..exceptions import *


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


@config.node
class Layer(Partition, classmap_entry="layer"):
    thickness = config.attr(type=float, required=_size_requirements)
    xz_scale = config.attr(
        type=types.or_(
            types.list(float, size=2), types.scalar_expand(float, lambda x: [x, x],),
        ),
        default=lambda: [1.0, 1.0],
        call_default=True,
    )
    xz_center = config.attr(type=bool, default=False)
    region = config.ref(region_ref, populate="partitions", required=True)
    z_index = config.attr(type=int, required=lambda s: "stack" in s)
    volume_scale = config.attr(type=float, required=_size_requirements)
    position = config.attr(type=types.list(type=float, size=3))
    volume_dimension_ratio = config.attr(type=types.list(type=float, size=3))
    scale_from_layers = config.reflist(layer_ref)

    # TODO: Layer stacking
    # TODO: Layer scaling
    # TODO: Layer centering
