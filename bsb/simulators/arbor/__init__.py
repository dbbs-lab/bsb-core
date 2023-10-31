import types

# fixme: Once this adapter is moved to bsb-arbor, arbor can be a hard dependency at import
#  time, and this code can be removed.
try:
    import arbor
except ImportError as e:
    import sys
    from bsb.reporting import warn

    errmsg = f"Can't use bsb-arbor: Arbor is not importable: {e}"
    warn(errmsg)

    class ArborMock(types.ModuleType):
        class recipe:
            def __getattr__(self, item):
                raise AttributeError(errmsg)

        def __getattr__(self, item):
            raise AttributeError(errmsg)

    sys.modules["arbor"] = ArborMock(name="arbor")

from bsb.simulation import SimulationBackendPlugin
from .simulation import ArborSimulation
from .adapter import ArborAdapter
from . import devices


__plugin__ = SimulationBackendPlugin(Simulation=ArborSimulation, Adapter=ArborAdapter)
