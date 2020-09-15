import abc, random, types
import numpy as np
from ..helpers import ConfigurableClass
from ..reporting import report
from ..exceptions import *
from time import time


class SimulatorAdapter(ConfigurableClass):
    def __init__(self):
        super().__init__()
        self.cell_models = {}
        self.connection_models = {}
        self.devices = {}
        self.entities = {}
        self._progress_listeners = []

    def get_configuration_classes(self):
        if not hasattr(self.__class__, "simulator_name"):
            raise AttributeMissingError(
                "The SimulatorAdapter {} is missing the class attribute 'simulator_name'".format(
                    self.__class__
                )
            )
        # Check for the 'configuration_classes' class attribute
        if not hasattr(self.__class__, "configuration_classes"):
            raise AdapterError(
                "The '{}' adapter class needs to set the 'configuration_classes' class attribute to a dictionary of configurable classes (str or class).".format(
                    self.simulator_name
                )
            )
        classes = self.configuration_classes
        keys = ["cell_models", "connection_models", "devices"]
        # Check for the presence of required classes
        for requirement in keys:
            if requirement not in classes:
                raise AdapterError(
                    "{} adapter: The 'configuration_classes' dictionary requires a class under the '{}' key.".format(
                        self.simulator_name, requirement
                    )
                )
        # Test if they are all children of the ConfigurableClass class
        for class_key in keys:
            if not issubclass(classes[class_key], ConfigurableClass):
                raise AdapterError(
                    "{} adapter: The configuration class '{}' should inherit from ConfigurableClass".format(
                        self.simulator_name, class_key
                    )
                )
        return self.configuration_classes

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
        progress = types.SimpleNamespace(
            progression=progression, duration=duration, time=time()
        )
        for listener in self._progress_listeners:
            listener(progress)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)
