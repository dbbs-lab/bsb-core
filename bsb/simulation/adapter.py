import abc


class SimulatorAdapter:
    def simulate(self, simulation):
        """
        Simulate the given simulation.
        """
        with simulation.scaffold.storage.read_only():
            data = self.prepare(simulation)
            for hook in simulation.post_prepare:
                hook(self, data)
            self.run(simulation)
            return self.collect(simulation, data)

    @abc.abstractmethod
    def prepare(self, simulation, comm=None):
        """
        Reset the simulation backend and prepare for the given simulations.

        :param simulation: The simulation configuration to prepare.
        :type simulation: ~bsb.simulation.simulation.Simulation
        :param comm: The MPI communicator to use. Only nodes in the communicator will
          participate in the simulation. The first node will idle as the main node. Calls
          :meth:`~bsb.simulation.adapter.SimulatorAdapter.set_communicator`
        """
        pass

    @abc.abstractmethod
    def run(self):
        """
        Fire up the prepared adapter.
        """
        pass

    @abc.abstractmethod
    def collect(self):
        """
        Collect the output of a simulation that completed
        """
        pass

    @abc.abstractmethod
    def set_communicator(self, comm):
        """
        Set the communicator for this adapter.
        """
        pass
