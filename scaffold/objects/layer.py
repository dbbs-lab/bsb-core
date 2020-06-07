"""
    Module for the Layer configuration node and its dependencies.
"""

from .. import config
from ..config import types
from ..config.refs import stack_ref


@config.node
class StackNode:
    stack = config.ref(stack_ref, required=True)
    z_index = config.attr(type=int, required=True)


@config.node
class Layer:
    name = config.attr(key=True)
    thickness = config.attr(type=float, required=True)
    xz_scale = config.attr(
        type=types.or_(
            types.list(float, size=2), types.scalar_expand(float, lambda x: [x, x],),
        ),
        default=lambda: [1.0, 1.0],
        call_default=True,
    )
    xz_center = config.attr(type=bool, default=False)
    stack = config.attr(type=StackNode)
