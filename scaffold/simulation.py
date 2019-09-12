import abc
import numpy as np
from .helpers import ConfigurableClass, assert_attr

class SimulatorAdapter(ConfigurableClass):

    def __init__(self):
        super().__init__()
        self.cell_types = {}
        self.connection_types = {}

    def get_configuration_classes(self):
        if not hasattr(self.__class__, 'simulator_name'):
            raise Exception("The SimulatorAdapter {} is missing the class attribute 'simulator_name'".format(self.__class__))
        # Check for the 'configuration_classes' class attribute
        if not hasattr(self.__class__, 'configuration_classes'):
            raise Exception("The '{}' adapter class needs to set the 'configuration_classes' class attribute to a dictionary of ConfigurableClass class objects.".format(self.simulator_name))
        classes = self.configuration_classes
        # Check for the presence of required classes
        if not 'cell_model' in classes:
            raise Exception("{} adapter: The 'configuration_classes' dictionary requires a class to handle the simulation configuration of cells under the 'cell_model' key.".format(self.simulator_name))
        if not 'connection' in classes:
            raise Exception("{} adapter: The 'configuration_classes' dictionary requires a class to handle the simulation configuration of cell connections under the 'connection' key.".format(self.simulator_name))
        # if not 'simulation' in classes:
        #     raise Exception("The 'configuration_classes' dictionary requires a class to handle the configuration of simulations under the 'simulation' key.")
        # Test if they are all children of the ConfigurableClass class
        keys = ['cell_model', 'connection'] #, 'simulation']
        for class_key in keys:
            if not issubclass(classes[class_key], ConfigurableClass):
                raise Exception("{} adapter: The configuration class '{}' should inherit from ConfigurableClass".format(self.simulator_name, class_key))
        return self.configuration_classes

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

    name = 'nest'

    defaults = {
        'synapse_model': 'static_synapse',
        'neuron_model': 'iaf'
    }

    required = ['neuron_model', 'synapse_model']

    def prepare(self, hdf5):
        import nest

        scaffold = self.scaffold
        configuration = scaffold.configuration
        self.create_neurons(configuration.cell_types)
        self.connect_neurons(configuration.connection_types, hdf5)
        self.stimulate_neurons()
        return nest

    def simulate(self, simulator):
        simulator.Simulate()

    def validate(self):
        pass

    def create_neurons(self, neuron_types):
        default_model = self.neuron_model
        for neuron_type in neuron_types.values():
            name = neuron_type.name
            nest_model_name = default_model
            if hasattr(neuron_type, "fixed_model"):
                nest_model_name = neuron_type.fixed_model
            nest.CopyModel(nest_model_name, name)
            nest.SetDefaults(name, cell_type.simulation.nest.models[nest_model_name])
            nest.Create(neuron_type.name, neuron_type.placement.cells_placed)

    def connect_neurons(self, connection_types, hdf5):
        default_model = self.synapse_model       # default model will be static_synapse
        for connection_type in connection_types.values():
            connectivity_matrix = hdf5['cells/connections'][connection_type.name]
            presynaptic_cells = connectivity_matrix[:,0]
            postsynaptic_cells = connectivity_matrix[:,1]
            synaptic_parameters = connection_type.simulation.nest.models[default_model]  # Dictionary with delay and weight
            connection_parameters = {'rule': 'one_to_one'}
            nest.Connect(presynaptic_cells, postsynaptic_cells, connection_parameters, synaptic_parameters)



    def stimulate_neurons(self):
        pass
