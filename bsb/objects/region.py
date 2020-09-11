from .. import config
from ..config import types

# Create a dynamic class that by default dynamically casts to this class.
@config.dynamic(required=False, default="bsb.objects.Region")
class Region:
    name = config.attr(key=True)
    origin = config.attr(type=types.list(type=float, size=3), required=True)

    def __boot__(self):
        pass
