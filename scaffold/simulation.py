import abc
import numpy as np
from .helpers import ConfigurableClass, assert_attr
from .models import NestCell, NestConnection, NestStimulus

class SimulatorAdapter(ConfigurableClass):

    def __init__(self):
        super().__init__()
        self.cell_models = {}
        self.connection_models = {}
        self.stimuli = {}

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
        if not 'stimuli' in classes:
            raise Exception("{} adapter: The 'configuration_classes' dictionary requires a class to handle the simulation configuration of stimuli under the 'stimuli' key.".format(self.simulator_name))

        # Test if they are all children of the ConfigurableClass class
        keys = ['cell_models', 'connection_models', 'stimuli']
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
        'connection_models': NestConnection,
        'stimuli': NestStimulus
    }

    defaults = {
        'default_synapse_model': 'static_synapse',
        'default_neuron_model': 'iaf'
    }

    required = ['default_neuron_model', 'default_synapse_model']

    def prepare(self, hdf5):
        import nest
        self.nest = nest
        self.create_neurons(self.cell_models)
        self.connect_neurons(self.connection_models, hdf5)
        self.stimulate_neurons({})
        return nest

    def simulate(self, simulator):
        simulator.Simulate(10)

    def validate(self):
        pass

    def create_neurons(self, cell_models):
        '''
            Recreate the scaffold neurons in the same order as they were placed,
            inside of the NEST simulator based on the cell model configuration.
        '''
        default_model = self.default_neuron_model
        # Iterate over all the placement stitches: each stitch was a batch of cells placed together and
        # if we don't follow the same order as during the placement, the cell IDs will not match
        for cell_type_id, start_id, count in self.scaffold.placement_stitching:
            # Get the cell_type name from the type id to type name map.
            name = self.scaffold.configuration.cell_type_map[cell_type_id]
            # Get the cell model
            cell_model = cell_models[name]
            # Use the default model unless another one is specified in the configuration.
            nest_model_name = cell_model.neuron_model if hasattr(cell_model, "neuron_model") else default_model
            # Alias the nest model name under our cell model name.
            self.nest.CopyModel(nest_model_name, name)
            # Get the synapse parameters
            params = cell_model.get_parameters(model=nest_model_name)
            # Set the parameters in NEST
            self.nest.SetDefaults(name, params)
            # Create the same amount of cells that were placed in this stitch.
            identifiers = self.nest.Create(name, count)
            # Check if the stitching is going OK.
            if identifiers[0] != start_id + 1:
                raise Exception("Could not match the scaffold cell identifiers to NEST identifiers! Cannot continue.")

    def connect_neurons(self, connection_models, hdf5):
        default_model = self.default_synapse_model
        for connection_model in connection_models.values():
            dataset_name = 'cells/connections/' + connection_model.name
            if not dataset_name in hdf5:
                if self.scaffold.configuration.verbosity > 0:
                    print('[WARNING] Expected connection dataset "{}" not found. Skipping it.'.format(dataset_name))
                continue
            connectivity_matrix = hdf5[dataset_name]
            # Translate the id's from 0 based scaffold ID's to NEST's 1 based ID's with '+ 1'
            presynaptic_cells = np.array(connectivity_matrix[:,0] + 1, dtype=int)
            postsynaptic_cells = np.array(connectivity_matrix[:,1] + 1, dtype=int)
            # Filter out these keys to use as parameters for the synapses
            parameter_keys = ['weight', 'delay']
            synaptic_parameters = {}
            for key in parameter_keys:
                if hasattr(connection_model, key):
                    synaptic_parameters[key] = connection_model.__dict__[key]
                else:
                    print('exluded:', key)
            connection_parameters = {'rule': 'one_to_one'}
            self.nest.Connect(presynaptic_cells, postsynaptic_cells, connection_parameters, synaptic_parameters)

    def stimulate_neurons(self, stimuli):
        for stimulus_model in stimuli.values():
            stimulus = self.nest.Create(stimulus_model.name)
            self.nest.SetStatus(stimulus, stimulus_model.parameters)
            self.nest.Connect(stimulus,stimulus_model.get_targets())
