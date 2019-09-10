import abc
import numpy as np
from .helpers import ConfigurableClass

class SimulatorAdapter(ConfigurableClass):
    @abc.abstractmethod
    def prepare(self, hdf5, simulation_config):
        '''
            This method turns a stored HDF5 network architecture and returns a runnable simulator.

            :returns: A simulator prepared to run a simulation according to the given configuration.
        '''
        pass

    @abc.abstractmethod
    def simulate(self, simulator):
        '''
            Start a simulation given a simulator object.
        '''
        pass

class NestAdapter(SimulatorAdapter):
    def prepare(self, hdf5, simulation_config):
        import nest

        scaffold = self.scaffold
        configuration = self.scaffold.configuration
        self.create_neurons(simulation_config,configuration.cell_types)
        self.connect_neurons(simulation_config, configuration.connection_types)
        self.stimulate_neurons()
        return nest

    def simulate(self, simulator):
        simulator.Simulate()

    def validate(self):
        pass

    def create_neurons(self,simulation_config, neuron_types):
        default_model = self.scaffold.simulation_config.neuron_model
        for neuron_type in neuron_types.values():
            name = neuron_type.name
            nest_model_name = default_model
            if hasattr(neuron_type, "fixed_model"):
                nest_model_name = neuron_type.fixed_model
            nest.CopyModel(nest_model_name, name)
            nest.SetDefaults(name, cell_type.simulation.nest.models[nest_model_name])
            # nest.Create(neuron_type.name,neuron_type.placement.number)

    def connect_neurons(self, simulation_config, connection_types):
        default_model = self.scaffold.simulation_config.synapse_model       # default model will be static_synapse
        for connection_type in connection_types.values():
            pre = connection_type.matrix[:,0]          # The connectivity matrix can be in the configuration file? Maybe better in the simulation configuration file?
            post = connection_type.matrix[:,1]
            syn_spec = connection_type.conn_parameters  # Dictionary with delay and weight
            conn_dict = {'rule': 'one_to_one'}
            nest.Connect(pre,post, conn_dict, syn_spec)
        
        

    def stimulate_neurons(self):
        pass
