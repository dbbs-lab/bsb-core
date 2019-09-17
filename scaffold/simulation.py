import abc
import numpy as np
from .helpers import ConfigurableClass, assert_attr
from .models import NestCell, NestConnection

class SimulatorAdapter(ConfigurableClass):

    def __init__(self):
        super().__init__()
        self.cell_models = {}
        self.connection_models = {}

    def get_configuration_classes(self):
        if not hasattr(self.__class__, 'simulator_name'):
            raise Exception("The SimulatorAdapter {} is missing the class attribute 'simulator_name'".format(self.__class__))
        # Check for the 'configuration_classes' class attribute
        if not hasattr(self.__class__, 'configuration_classes'):
            raise Exception("The '{}' adapter class needs to set the 'configuration_classes' class attribute to a dictionary of configurable classes (str or class).".format(self.simulator_name))
        classes = self.configuration_classes
        # Check for the presence of required classes
        if not 'cell_models' in classes:
            raise Exception("{} adapter: The 'configuration_classes' dictionary requires a class to handle the simulation configuration of cells under the 'cell_models' key.".format(self.simulator_name))
        if not 'connection_models' in classes:
            raise Exception("{} adapter: The 'configuration_classes' dictionary requires a class to handle the simulation configuration of cell connections under the 'connection_models' key.".format(self.simulator_name))
        # if not 'simulation' in classes:
        #     raise Exception("The 'configuration_classes' dictionary requires a class to handle the configuration of simulations under the 'simulation' key.")
        # Test if they are all children of the ConfigurableClass class
        keys = ['cell_models', 'connection_models'] #, 'simulation']
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

    simulator_name = 'nest'

    configuration_classes = {
        'cell_models': NestCell,
        'connection_models': NestConnection
    }

    defaults = {
        'synapse_model': 'static_synapse',
        'neuron_model': 'iaf'
    }

    required = ['neuron_model', 'synapse_model']

    def prepare(self, hdf5):
        import nest
        self.nest = nest
        self.create_neurons(self.cell_models)
        self.connect_neurons(self.connection_models, hdf5)
        self.stimulate_neurons({})
        return nest

    def simulate(self, simulator):
        simulator.Simulate()

    def validate(self):
        pass

    def create_neurons(self, cell_models):
        default_model = self.default_neuron_model
        for cell_model in cell_models.values():
            name = cell_model.name
            nest_model_name = cell_model.neuron_model if hasattr(cell_model, "neuron_model") else default_model
            self.nest.CopyModel(nest_model_name, name)
            params = cell_model.parameters.copy()
            if not hasattr(cell_model, nest_model_name):
                raise Exception("Missing parameters for '{}' model in '{}'".format(nest_model_name, name))
            params.update(cell_model.__dict__[nest_model_name])
            self.nest.SetDefaults(name, params)
            self.handle = self.nest.Create(cell_model.name, self.scaffold.statistics.cells_placed[name])

    def connect_neurons(self, connection_models, hdf5):
        default_model = self.default_synapse_model
        for connection_model in connection_models.values():
            connectivity_matrix = hdf5['cells/connections'][connection_model.name]
            presynaptic_cells = np.array(connectivity_matrix[:,0], dtype=int)
            postsynaptic_cells = np.array(connectivity_matrix[:,1], dtype=int)
            parameter_keys = ['weight', 'delay']
            synaptic_parameters = {}
            for key in parameter_keys:
                if hasattr(connection_model, key):
                    synaptic_parameters[key] = connection_model.__dict__[key]
            connection_parameters = {'rule': 'one_to_one'}
            self.nest.Connect(presynaptic_cells, postsynaptic_cells, connection_parameters, synaptic_parameters)

    def stimulate_neurons(self, stimulus_models):
        for stimulus_model in stimulus_models.values():
            stimulus = self.nest.Create(stimulus_model.name)
            self.nest.SetStatus(stimulus, stimulus_model.parameters)
            self.nest.Connect(stimulus,stimulus_model.get_targets())
