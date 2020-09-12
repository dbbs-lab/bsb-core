from .. import config
from ..config import types, refs


@config.dynamic(required=False, default="layer_stack", auto_classmap=True)
class Region:
    name = config.attr(key=True)
    origin = config.attr(
        type=types.list(type=float, size=3),
        default=lambda: [0.0, 0.0, 0.0],
        call_default=True,
    )
    partitions = config.reflist(refs.regional_ref)

    def __boot__(self):
        pass


@config.node
class LayerStack(Region, classmap_entry="layer_stack"):
    pass
