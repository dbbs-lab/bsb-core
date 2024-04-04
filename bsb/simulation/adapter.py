import abc
import itertools
import types
import typing
from contextlib import ExitStack
from time import time

import numpy as np

from .results import SimulationResult

if typing.TYPE_CHECKING:
    from ..storage import PlacementSet
    from .cell import CellModel
    from .simulation import Simulation


class AdapterProgress:
    def __init__(self, duration):
        self._duration = duration
        self._start = self._last_tick = time()
        self._ticks = 0

    def tick(self, step):
        """
        Report simulation progress.
        """
        now = time()
        tic = now - self._last_tick
        self._ticks += 1
        el = now - self._start
        progress = types.SimpleNamespace(
            progression=step, duration=self._duration, time=time(), tick=tic, elapsed=el
        )
        self._last_tick = now
        return progress

    def steps(self, step=1):
        steps = itertools.chain(np.arange(0, self._duration, step), (self._duration,))
        a, b = itertools.tee(steps)
        next(b, None)
        yield from zip(a, b)

    def complete(self):
        return


class SimulationData:
    def __init__(self, simulation: "Simulation", result=None):
        self.chunks = None
        self.populations = dict()
        self.placement: dict["CellModel", "PlacementSet"] = {
            model: model.get_placement_set() for model in simulation.cell_models.values()
        }
        self.connections = dict()
        self.devices = dict()
        if result is None:
            result = SimulationResult(simulation)
        self.result: SimulationResult = result


class SimulatorAdapter(abc.ABC):
    def __init__(self):
        self._progress_listeners = []
        self.simdata: dict["Simulation", "SimulationData"] = dict()

    def simulate(self, *simulations, post_prepare=None, comm=None):
        """
        Simulate the given simulations.
        """
        with ExitStack() as context:
            for simulation in simulations:
                context.enter_context(simulation.scaffold.storage.read_only())
            alldata = []
            for simulation in simulations:
                data = self.prepare(simulation)
                alldata.append(data)
                for hook in simulation.post_prepare:
                    hook(self, simulation, data)
            if post_prepare:
                post_prepare(self, simulations, alldata)
            results = self.run(*simulations)
            return [
                self.collect(simulation, data, result)
                for simulation, result in zip(simulations, results)
            ]

    @abc.abstractmethod
    def prepare(self, simulation, comm=None):
        """
        Reset the simulation backend and prepare for the given simulation.

        :param simulation: The simulation configuration to prepare.
        :type simulation: ~bsb.simulation.simulation.Simulation
        :param comm: The mpi4py MPI communicator to use. Only nodes in the communicator
          will participate in the simulation. The first node will idle as the main node.
        """
        pass

    @abc.abstractmethod
    def run(self, *simulations, comm=None):
        """
        Fire up the prepared adapter.
        """
        pass

    def collect(self, simulation, simdata, simresult, comm=None):
        """
        Collect the output of a simulation that completed
        """
        simresult.flush()
        return simresult

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)


__all__ = ["AdapterProgress", "SimulationData", "SimulatorAdapter"]
