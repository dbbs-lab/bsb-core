from ..simulation import SimulatorAdapter, SimulationComponent
import numpy as np
from sklearn.neighbors import KDTree

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
        "synapse": dict,
        "connection": dict
    }

    required = ["synapse", "connection"]

    defaults = {
        'plastic': False,
        'hetero': None,
        'teaching': None,
    }

    def validate(self):
        if self.plastic:
            # Set plasticity synapse dict defaults
            synapse_defaults = {
                "A_minus": 0.0,
                "A_plus": 0.0,
                "Wmin": 0.0,
                "Wmax": 4000.0
            }
            for key, value in synapse_defaults.items():
                if not key in self.synapse:
                    self.synapse[key] = value

    def get_synapse_parameters(self):
        # Get the default synapse parameters
        return self.synapse


    def get_connection_parameters(self, default_model):
        # Use the default model unless another one is specified in the configuration.
        nest_synapse_name = self.synapse_model if hasattr(self, "synapse_model") else default_model
        # Get the default synapse parameters
        params = self.connection["parameters"].copy()
        # Raise an exception if the requested model is not configured.
        if not nest_synapse_name in self.connection:
            raise Exception("Missing connection parameters for '{}' model in '{}'".format(nest_synapse_name, self.name + '.connection'))
        # Merge in the model specific parameters
        params.update(self.connection[nest_synapse_name])
        return params


class NestDevice(SimulationComponent):
    node_name = 'simulations.?.devices'

    casts = {
        'radius': float,
        'origin': [float],
        'parameters': dict
    }

    required = ['type', 'device', 'io', 'parameters']

    def validate(self):
        # Fill in the _get_targets method, so that get_target functions
        # according to `type`.
        types = ['local', 'cell_type']
        if not self.type in types:
            raise Exception("Unknown NEST targetting type '{}' in {}".format(self.type, self.node_name))
        get_targets_name = '_targets_' + self.type
        method = getattr(self, get_targets_name) if hasattr(self, get_targets_name) else None
        if not callable(method):
            raise Exception("Unimplemented NEST stimulation type '{}' in {}".format(self.type, self.node_name))
        self._get_targets = method
        if not self.io == "input" and not self.io == "output":
            raise Exception("Attribute io needs to be either 'input' or 'output' in {}".format(self.node_name))

    def get_targets(self):
        '''
            Return the targets of the stimulation to pass into the nest.Connect call.
        '''
        return (np.array(self._get_targets(), dtype=int) + 1).tolist()

    def _targets_local(self):
        '''
            Target all or certain cells in a spherical location.
        '''
        if len(self.cell_types) != 1:
            # Compile a list of the cells and build a compound tree.
            target_cells = np.empty((0, 5))
            id_map = np.empty((0,1))
            for t in self.cell_types:
                cells = self.scaffold.get_cells_by_type(t)
                target_cells = np.vstack((target_cells, cells[:, 2:5]))
                id_map = np.vstack((id_map, cells[:, 0]))
            tree = KDTree(target_cells)
            target_positions = target_cells
        else:
            # Retrieve the prebuilt tree from the SHDF file
            tree = self.scaffold.trees.cells.get_tree(self.cell_types[0])
            target_cells = self.scaffold.get_cells_by_type(self.cell_types[0])
            id_map = target_cells[:, 0]
            target_positions = target_cells[:, 2:5]
        # Query the tree for all the targets
        target_ids = tree.query_radius(np.array(self.origin).reshape(1, -1), self.radius)[0].tolist()
        return id_map[target_ids]

    def _targets_cell_type(self):
        '''
            Target all cells of certain cell types
        '''
        if len(self.cell_types) != 1:
            # Compile a list of the different cell type cells.
            target_cells = np.empty((0, 1))
            for t in self.cell_types:
                cells_of_type = self.scaffold.get_cells_by_type(t)
                target_cells = np.vstack((target_cells, cells_of_type[:, 0]))
            return target_cells
        else:
            # Retrieve a single list
            cells = self.scaffold.get_cells_by_type(self.cell_types[0])
            return cells[:, 0]


class NestAdapter(SimulatorAdapter):
    '''
        Interface between the scaffold model and the NEST simulator.
    '''

    simulator_name = 'nest'

    configuration_classes = {
        'cell_models': NestCell,
        'connection_models': NestConnection,
        'devices': NestDevice
    }

    casts = {
        'threads': int,
        'virtual_processes': int
    }

    defaults = {
        'default_synapse_model': 'static_synapse',
        'default_neuron_model': 'iaf',
        'verbosity': 'M_ERROR',
        'threads': 1,
        'virtual_processes': 1
    }

    required = ['default_neuron_model', 'default_synapse_model', 'duration']

    def __init__(self):
        super().__init__()

    def prepare(self, hdf5):
        import nest
        nest.set_verbosity(self.verbosity)
        nest.ResetKernel()
        nest.SetKernelStatus({
            'local_num_threads': self.threads,
            'total_num_virtual_procs': self.virtual_processes,
            'overwrite_files': True,
            'data_path': self.scaffold.output_formatter.get_simulator_output_path(self.simulator_name)
        })
        self.nest = nest
        self.create_neurons(self.cell_models)
        self.connect_neurons(self.connection_models, hdf5)
        self.create_devices(self.devices)
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
        track_models = [] # Keeps track of already added models if there's more than 1 stitch per model
        # Iterate over all the placement stitches: each stitch was a batch of cells placed together and
        # if we don't follow the same order as during the placement, the cell IDs can not be easily matched
        for cell_type_id, start_id, count in self.scaffold.placement_stitching:
            # Get the cell_type name from the type id to type name map.
            name = self.scaffold.configuration.cell_type_map[cell_type_id]
            if name not in track_models: # Is this the first time encountering this model?
                # Create the cell model in the simulator
                self.create_model(cell_models[name])
                track_models.append(name)
            # Create the same amount of cells that were placed in this stitch.
            identifiers = self.nest.Create(name, count)
            # Check if the stitching is going OK.
            if identifiers[0] != start_id + 1:
                raise Exception("Could not match the scaffold cell identifiers to NEST identifiers! Cannot continue.")

    def connect_neurons(self, connection_models, hdf5):
        '''
            Connect the cells in NEST according to the connection model configurations
        '''
        # TODO: with CopyModels()!!!! And SetDefaults()!!!!
        default_model = self.default_synapse_model
        track_models = [] # Keeps track of already added models if there'smodel=synapse_model more than 1 stitch per model
        for connection_model in connection_models.values():
            name = connection_model.name
            dataset_name = 'cells/connections/' + name
            if not dataset_name in hdf5:
                if self.scaffold.configuration.verbosity > 0:
                    print('[WARNING] Expected connection dataset "{}" not found. Skipping it.'.format(dataset_name))
                continue
            connectivity_matrix = hdf5[dataset_name]
            # Transdefault_modellate the id's from 0 based scaffold ID's to NEST's 1 based ID's with '+ 1'
            presynaptic_cells = np.array(connectivity_matrix[:,0] + 1, dtype=int)
            postsynaptic_cells = np.array(connectivity_matrix[:,1] + 1, dtype=int)
            if name not in track_models: # Is this the first time encountering this model?
                # Create the cell model in the simulator
                self.create_synapse_model(connection_model, default_model)
                if synapse_model.plastic == True:
                    self.create_volume_transmitter(len(postsynaptic_cells))
                track_models.append(name)

            # Set the specifications NEST allows like: 'rule', 'autapses', 'multapses'
            connection_specifications = {'rule': 'one_to_one'}
            connection_parameters = connection_model.get_connection_parameters(default_model=default_model)
            # Create the connections in NEST
            self.nest.Connect(presynaptic_cells, postsynaptic_cells, connection_specifications, connection_parameters)

            # Associate volume transmitter if plastic
            if synapse_model.plastic == True:
                for i,tar in enumerate(postsynaptic_cells):
                    A=nest.GetConnections(presynaptic_cells,[tar])
                    nest.SetStatus(A,{'n': float(i)})


    def create_devices(self, devices):
        '''
            Create the configured NEST devices in the simulator
        '''
        for device_model in devices.values():
            device = self.nest.Create(device_model.device)
            device_targets = device_model.get_targets()
            self.nest.SetStatus(device, device_model.parameters)
            if device_model.io == "input":
                self.nest.Connect(device, device_targets)
            else:
                self.nest.Connect(device_targets, device)

    def create_model(self, cell_model):
        '''
            Create a NEST cell model in the simulator based on a cell model configuration.
        '''
        # Use the default model unless another one is specified in the configuration.A_minus
        nest_model_name = cell_model.neuron_model if hasattr(cell_model, "neuron_model") else self.default_neuron_model
        # Alias the nest model name under our cell model name.
        self.nest.CopyModel(nest_model_name, cell_model.name)
        # Get the synapse parameters
        params = cell_model.get_parameters(model=nest_model_name)
        # Set the parameters in NEST
        self.nest.SetDefaults(cell_model.name, params)

    def create_synapse_model(self, synapse_model, default_model):
        '''
            Create a NEST synapse model in the simulator based on a synapse model configuration.
        '''
        # Create volume transmitter if it is plastic
        if synapse_model.plastic == True:
            vt = nest.Create("volume_transmitter_alberto",num_targets)

            # Set vt get_parameters
            for n,vti in enumerate(vt):
        		nest.SetStatus([vti],{"deliver_interval" : 2})            # TO CHECK
        		nest.SetStatus([vti],{"n" : n})

        # Use the default model unless another one is specified in the configuration.
        nest_synapse_name = synapse_model.synapse_model if hasattr(synapse_model, "synapse_model") else self.default_synapse_model
        # Alias the nest model name under our cell model name.
        self.nest.CopyModel(nest_synapse_name, synapse_model.name)
        # Get the synapse parameters
        params = synapse_model.get_synapse_parameters()
        # Set the parameters in NEST
        self.nest.SetDefaults(synapse_model.name, params)

        # Create volume transmitter if it is plastic
    def create_volume_transmitter(self, len_target, nest_synapse_name):
        vt = nest.Create("volume_transmitter_alberto",len_target)

        nest.SetDefaults(nest_synapse_name,{"vt":   vt[0]}

            # vt properties
        for n,vti in enumerate(vt):
        	nest.SetStatus([vti],{"deliver_interval" : 2})            # TO CHECK
        	nest.SetStatus([vti],{"n" : n})
