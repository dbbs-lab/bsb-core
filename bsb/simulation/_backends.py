import functools
import typing

from .. import plugins

if typing.TYPE_CHECKING:
    from . import SimulationBackendPlugin
    from .adapter import SimulatorAdapter
    from .simulation import Simulation


@functools.cache
def get_backends() -> dict[str, "SimulationBackendPlugin"]:
    backends = plugins.discover("simulation_backends")
    for backend in backends.values():
        plugins._decorate_advert(backend.Simulation, backend._bsb_entry_point)
        plugins._decorate_advert(backend.Adapter, backend._bsb_entry_point)
    return backends


@functools.cache
def get_simulation_nodes() -> dict[str, "Simulation"]:
    return {name: plugin.Simulation for name, plugin in get_backends().items()}


@functools.cache
def get_simulation_adapters() -> dict[str, "SimulatorAdapter"]:
    return {name: plugin.Adapter() for name, plugin in get_backends().items()}
