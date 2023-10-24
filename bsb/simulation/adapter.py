import abc


class SimulatorAdapter(abc.ABC):
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
