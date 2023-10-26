import abc
import typing
from .results import SimulationResult


if typing.TYPE_CHECKING:
    from .simulation import Simulation
    from .cell import CellModel
    from ..storage import PlacementSet


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
        self.simdata: dict["Simulation", "SimulationData"] = dict()

    def simulate(self, simulation):
        """
        Simulate the given simulation.
        """
        with simulation.scaffold.storage.read_only():
            data = self.prepare(simulation)
            for hook in simulation.post_prepare:
                hook(self, simulation, data)
            result = self.run(simulation)
            return self.collect(simulation, data, result)

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
    def run(self, simulation):
        """
        Fire up the prepared adapter.
        """
        pass

    def collect(self, simulation, simdata, simresult):
        """
        Collect the output of a simulation that completed
        """
        simresult.flush()
        return simresult
