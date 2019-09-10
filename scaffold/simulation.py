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
	'''
		Interface between the scaffold model and the NEST simulator.
	'''

	defaults = {
		'synapse_model': 'static_synapse'
	}

	required = ['neuron_model', 'synapse_model']

    def prepare(self, hdf5):
        import nest

        scaffold = self.scaffold
        configuration = self.scaffold.configuration
        self.create_neurons(configuration.cell_types)
        self.connect_neurons(configuration.connection_types, hdf5)
        self.stimulate_neurons()
        return nest

    def simulate(self, simulator):
        simulator.Simulate()

    def validate(self):
        pass

    def create_neurons(neuron_types):
        default_model = self.neuron_model
        for neuron_type in neuron_types.values():
            name = neuron_type.name
            nest_model_name = default_model
            if hasattr(neuron_type, "fixed_model"):
                nest_model_name = neuron_type.fixed_model
            nest.CopyModel(nest_model_name, name)
            nest.SetDefaults(name, cell_type.simulation.nest.models[nest_model_name])
            # nest.Create(neuron_type.name,neuron_type.placement.number)

    def connect_neurons(self, connection_types, hdf5):
        default_model = self.synapse_model       # default model will be static_synapse
        for connection_type in connection_types.values():
            connectivity_matrix = hdf5['connections'][connection_type.name]
            presynaptic_cells = connectivity_matrix[:,0]
            postsynaptic_cells = connectivity_matrix[:,1]
            synaptic_parameters = connection_type.simulation.nest.models[default_model]  # Dictionary with delay and weight
            connection_parameters = {'rule': 'one_to_one'}
            nest.Connect(presynaptic_cells, postsynaptic_cells, connection_parameters, synaptic_parameters)



    def stimulate_neurons(self):
        pass
