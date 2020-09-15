import abc, random, types
import numpy as np
from ..reporting import report
from ..exceptions import *
from time import time
from .. import plugins, config
from .cell import CellModel
from .connection import ConnectionModel
from .device import DeviceModel


class ProgressEvent:
    def __init__(progression, duration, time):
        self.progression = progression
        self.duration = duration
        self.time = time


@config.pluggable(key="simulator", plugin_name="simulator adapter")
class SimulatorAdapter:
    duration = config.attr(type=float, required=True)
    cell_models = config.slot(type=CellModel, required=True)
    connection_models = config.slot(type=ConnectionModel, required=True)
    devices = config.slot(type=DeviceModel, required=True)

    @classmethod
    def __plugins__(cls):
        if not hasattr(cls, "_plugins"):
            cls._plugins = plugins.discover("adapters")
        return cls._plugins

    def __init__(self):
        self.entities = {}
        self._progress_listeners = []

    @abc.abstractmethod
    def prepare(self, hdf5, simulation_config):
        """
            This method turns a stored HDF5 network architecture and returns a runnable simulator.

            :returns: A simulator prepared to run a simulation according to the given configuration.
        """
        pass

    @abc.abstractmethod
    def simulate(self, simulator):
        """
            Start a simulation given a simulator object.
        """
        pass

    @abc.abstractmethod
    def collect_output(self, simulator):
        """
            Collect the output of a simulation that completed
        """
        pass

    def progress(self, progression, duration):
        report("Simulated {}/{}ms".format(progression, duration), level=3, ongoing=True)
        progress = ProgressEvent(progression, duration, time())
        for listener in self._progress_listeners:
            listener(progress)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)
