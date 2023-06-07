from .. import plugins
import functools


@functools.cache
def get_backends():
    backends = plugins.discover("simulation_backends")
    for backend in backends.values():
        plugins._decorate_advert(backend.Simulation, backend._bsb_entry_point)
        plugins._decorate_advert(backend.Adapter, backend._bsb_entry_point)
    return backends


@functools.cache
def get_simulation_nodes():
    return {name: plugin.Simulation for name, plugin in get_backends().items()}


@functools.cache
def get_simulation_adapters():
    return {name: plugin.Adapter() for name, plugin in get_backends().items()}
