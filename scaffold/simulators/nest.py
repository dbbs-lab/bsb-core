from ..simulation import SimulatorAdapter, SimulationComponent
import numpy as np

class NestCell(SimulationComponent):

    node_name = 'simulations.?.cell_models'
    required = ['parameters']

    def validate(self):
        pass

    def get_parameters(self, model=None):
        # Get the default synapse parameters
        params = self.parameters.copy()
        # If a model is specified, fetch model specific parameters
        if not model is None:
            # Raise an exception if the requested model is not configured.
            if not hasattr(self, model):
                raise Exception("Missing parameters for '{}' model in '{}'".format(model, self.name))
            # Merge in the model specific parameters
            params.update(self.__dict__[model])
        return params

class NestConnection(SimulationComponent):
    node_name = 'simulations.?.connection_models'

    casts = {
        'weight': float,
        'delay': float
    }

    required = ['weight', 'delay']

    def validate(self):
        pass

class NestStimulus(SimulationComponent):
    node_name = 'simulations.?.stimuli'

    casts = {
        'radius': float,
        'origin': [float]
    }

    required = ['type', 'device']

    def validate(self):
        # Replace the get_stimulation method by the stimulate_<type> method, so that get_stimulation always
        # stimulates according to type.
        types = ['local']
        if not self.type in types:
            raise Exception("Unknown NEST stimulation type '{}' in {}".format(self.type, self.node_name))
        get_targets_name = '_targets_' + self.type
        method = getattr(self, get_targets_name) if hasattr(self, get_targets_name) else None
        if not callable(method):
            raise Exception("Unimplemented NEST stimulation type '{}' in {}".format(self.type, self.node_name))
        self.get_targets = method

    def get_targets(self):
        '''
            *This method gets replaced by one of the _targets_x methods, depending on the value of self.type.*

            Return the targets of the stimulation to pass into the nest.Connect call.
        '''
        pass

    def _targets_local(self):
        target_type = self.scaffold.get_cell_type(name=self.targets.types)
        tree = kdtree(target_cells)
        target_ids = tree.query_radius(self.origin, self.radius)
        return target_ids


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

    required = ['default_neuron_model', 'default_synapse_model', 'duration']

    def prepare(self, hdf5):
        import nest
        self.nest = nest
        self.create_neurons(self.cell_models)
        self.connect_neurons(self.connection_models, hdf5)
        self.stimulate_neurons(self.stimuli)
        return nest

    def simulate(self, simulator):
        simulator.Simulate(self.duration)

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
            # Filter the parameter keys from the connection_model
            parameter_keys = ['weight', 'delay']
            synaptic_parameters = {}
            for key in parameter_keys:
                if hasattr(connection_model, key):
                    synaptic_parameters[key] = connection_model.__dict__[key]

            # Set the specifications NEST allows like: 'rule', 'autapses', 'multapses'
            connection_specifications = {'rule': 'one_to_one'}
            # Create the connections in NEST
            self.nest.Connect(presynaptic_cells, postsynaptic_cells, connection_specifications, synaptic_parameters)

    def stimulate_neurons(self, stimuli):
        for stimulus_model in stimuli.values():
            stimulus = self.nest.Create(stimulus_model.device)
            self.nest.SetStatus(stimulus, stimulus_model.parameters)
            self.nest.Connect(stimulus, stimulus_model.get_targets())
