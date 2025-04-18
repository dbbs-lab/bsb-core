import abc
import itertools
import types
import typing
from contextlib import ExitStack
from time import time

import numpy as np

from ..services.mpi import MPIService
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


class AdapterCheckpoint:
    """Class that manages checkpointing of a simulation. In self.checkpoints a dictionary is saved with the checkpoint time as key and the value
    is a list of simulations that have to flush at that checkpoint.
    The get_status() method should be called in SimulatorAdapter run() to check if a checkpoint is reached.
    """

    def __init__(self, simulations):
        self.simulations = simulations
        self.resolutions = []
        self.checkpoints = {}
        for sim in simulations:
            for device in sim.devices.values():
                self.resolutions.append(sim.resolution)
                device_ckp = device.get_checkpoints(sim.duration, sim.resolution)
                for checkpoint in device_ckp:
                    if checkpoint not in self.checkpoints:
                        self.checkpoints[checkpoint] = [sim]
                    else:
                        self.checkpoints[checkpoint].append(sim)
        self.iterator = iter(self.sort_checkpoints())
        self.status = next(self.iterator, None)

    def sort_checkpoints(self):
        return sorted(self.checkpoints.keys())

    def get_status(self, i):
        """Checks whether the current simulation time has reached a checkpoint. If so, it advances to the next checkpoint"""
        if self.status == i:
            self.status = next(self.iterator, None)
            return True
        else:
            return False

    def suitable_step(self, pstep):
        """
        Check the greatest common divisor between progression step (pstep) and checkpoints value.

        :return: gdc interval (float).
        """
        sorted = np.array(self.sort_checkpoints())
        max_resolution = max(self.resolutions)
        if pstep == int(pstep):
            check_multiple = sorted % pstep
        else:
            check_multiple = sorted / pstep - np.array(sorted / pstep, dtype=int)
        if all(check_multiple == 0):
            return pstep
        elif any(sorted / max_resolution != np.array(sorted / max_resolution, dtype=int)):
            raise ValueError(
                f"Provided checkpoints are not multiple of resolution: {max_resolution}"
            )
        else:
            # We are here because pstep is too large. Now we look for the GDC between pstep and our checkpoints
            converted = np.array(sorted / max_resolution, dtype=int)
            min_step = int(pstep / max_resolution)
            for i in range(0, len(converted)):
                min_step = np.gcd(min_step, converted[i])
            return min_step * max_resolution


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
    def __init__(self, comm=None):
        """
        :param comm: The mpi4py MPI communicator to use. Only nodes in the communicator
          will participate in the simulation. The first node will idle as the main node.
        """
        self._progress_listeners = []
        self.simdata: dict["Simulation", "SimulationData"] = dict()
        self.comm = MPIService(comm)

    def simulate(self, *simulations, post_prepare=None):
        """
        Simulate the given simulations.

        :param simulations: One or a list of simulation configurations to simulate.
        :type simulations: ~bsb.simulation.simulation.Simulation
        :param post_prepare: Optional callable to run after the simulations' preparation.
        :return: List of simulation results for each simulation run.
        :rtype: list[~bsb.simulation.results.SimulationResult]
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
            return self.collect(results)

    @abc.abstractmethod
    def prepare(self, simulation):
        """
        Reset the simulation backend and prepare for the given simulation.

        :param simulation: The simulation configuration to prepare.
        :type simulation: ~bsb.simulation.simulation.Simulation
        :return: Prepared simulation data.
        :rtype: SimulationData
        """
        pass

    @abc.abstractmethod
    def run(self, *simulations):
        """
        Fire up the prepared adapter.

        :param simulations: One or a list of simulation configurations to simulate.
        :type simulations: ~bsb.simulation.simulation.Simulation
        :return: List of simulation results.
        :rtype: list[~bsb.simulation.results.SimulationResult]
        """
        pass

    def collect(self, results):
        """
        Collect the output the simulations that completed

        :return: Collected simulation results.
        :rtype: list[~bsb.simulation.results.SimulationResult]
        """
        for result in results:
            result.flush()
        return results

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)


__all__ = ["AdapterCheckpoint", "AdapterProgress", "SimulationData", "SimulatorAdapter"]
