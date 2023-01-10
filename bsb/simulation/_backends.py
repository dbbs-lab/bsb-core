from .. import plugins
import functools


@functools.cache
def get_backends():
    return plugins.discover("simulation_backends")


@functools.cache
def get_simulation_nodes():
    return {name: plugin.Simulation for name, plugin in get_backends().items()}


@functools.cache
def get_simulation_adapters():
    return {name: plugin.Adapter() for name, plugin in get_backends().items()}
