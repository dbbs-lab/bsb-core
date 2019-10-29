from ..simulation import SimulatorAdapter, SimulationComponent
import numpy as np
from sklearn.neighbors import KDTree

class NestCell(SimulationComponent):

    node_name = 'simulations.?.cell_models'
    required = ['parameters']

    def boot(self):
        self.identifiers = []
        self.receptor_specifications = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if key != "parameters" and isinstance(value, dict) and "receptors" in value:
                self.receptor_specifications[key] = value["receptors"]
                del value["receptors"]

    def validate(self):
        pass

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        # Raise an exception if the requested model is not configured.
        if not hasattr(self, self.neuron_model):
            raise Exception("Missing parameters for '{}' model in '{}'".format(self.neuron_model, self.name))
        # Merge in the model specific parameters
        params.update(self.__dict__[self.neuron_model])
        return params

    def get_receptor_specifications(self):
        return self.receptor_specifications[self.neuron_model] if self.neuron_model in self.receptor_specifications else {}

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


    def get_connection_parameters(self):
        # Get the default synapse parameters
        params = self.connection["parameters"].copy()
        # Raise an exception if the requested model is not configured.
        if not self.synapse_model in self.connection:
            raise Exception("Missing connection parameters for '{}' model in '{}'".format(self.synapse_model, self.name + '.connection'))
        # Merge in the model specific parameters
        params.update(self.connection[self.synapse_model])
        if self.should_specify_receptor_type():
            params["receptor_type"] = self.get_receptor_type()
        params["model"] = self.synapse_model
        return params

    def _get_cell_types(self, key="from"):
        meta = self.scaffold.output_formatter.get_connectivity_set_meta(self.name)
        if key + '_cell_types' in meta:
            cell_types = set()
            for name in meta[key + '_cell_types']:
                cell_types.add(self.scaffold.get_cell_type(name))
            return list(cell_types)
        connection_types = self.scaffold.output_formatter.get_connectivity_set_connection_types(self.name)
        cell_types = set()
        for connection_type in connection_types:
            cell_types |= set(connection_type.__dict__[key + "_cell_types"])
        return list(cell_types)

    def get_cell_types(self):
        return self._get_cell_types(key="from"), self._get_cell_types(key="to")

    def should_specify_receptor_type(self):
        _, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError("Specifying receptor types of connections consisiting of more than 1 cell type is currently undefined behaviour.")
        to_cell_type = to_cell_types[0]
        to_cell_model = self.adapter.cell_models[to_cell_type.name]
        return to_cell_model.neuron_model in to_cell_model.receptor_specifications

    def get_receptor_type(self):
        from_cell_types, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError("Specifying receptor types of connections consisiting of more than 1 target cell type is currently undefined behaviour.")
        if len(from_cell_types) > 1:
            raise NotImplementedError("Specifying receptor types of connections consisting of more than 1 origin cell type is currently undefined behaviour.")
        to_cell_type = to_cell_types[0]
        from_cell_type = from_cell_types[0]
        to_cell_model = self.adapter.cell_models[to_cell_type.name]
        from_cell_model = self.adapter.cell_models[from_cell_type.name]
        receptors = to_cell_model.get_receptor_specifications()
        if not from_cell_model.name in receptors:
            raise Exception("Missing receptor specification for cell model '{}' in '{}' while attempting to connect a '{}' to it during '{}'".format(
                to_cell_model.name,
                self.node_name,
                from_cell_model.name,
                self.name
            ))
        return receptors[from_cell_model.name]

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
        'virtual_processes': int,
        'modules': list
    }

    defaults = {
        'default_synapse_model': 'static_synapse',
        'default_neuron_model': 'iaf_cond_alpha',
        'verbosity': 'M_ERROR',
        'threads': 1,
        'virtual_processes': 1,
        'modules': []
    }

    required = ['default_neuron_model', 'default_synapse_model', 'duration']

    def __init__(self):
        super().__init__()

    def prepare(self, hdf5):
        import nest
        self.nest = nest
        nest.set_verbosity(self.verbosity)
        nest.ResetKernel()
        self.install_modules()
        nest.SetKernelStatus({
            'local_num_threads': self.threads,
            'total_num_virtual_procs': self.virtual_processes,
            'overwrite_files': True,
            'data_path': self.scaffold.output_formatter.get_simulator_output_path(self.simulator_name)
        })
        self.scaffold.report("Creating neurons...",2)
        self.create_neurons(self.cell_models)
        self.scaffold.report("Creating connections...",2)
        self.connect_neurons(self.connection_models, hdf5)
        self.scaffold.report("Creating devices...",2)
        self.create_devices(self.devices)
        return nest

    def simulate(self, simulator):
        simulator.Simulate(self.duration)
        self.scaffold.report("Simulating...",2)

    def validate(self):
        for cell_model in self.cell_models.values():
            cell_model.neuron_model = cell_model.neuron_model if hasattr(cell_model, "neuron_model") else self.default_neuron_model
        for connection_model in self.connection_models.values():
            connection_model.synapse_model = connection_model.synapse_model if hasattr(connection_model, "synapse_model") else self.default_synapse_model

    def install_modules(self):
        for module in self.modules:
            try:
                self.nest.Install(module)
            except Exception as e:
                if e.errorname == "DynamicModuleManagementError":
                    self.scaffold.report("[WARNING] Module {} already installed".format(module),1)
                else:
                    raise


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
                self.scaffold.report("Creating "+name+"...", 3)
                self.create_model(cell_models[name])
                track_models.append(name)
            # Create the same amount of cells that were placed in this stitch.
            self.scaffold.report("Creating {} {}...".format(count, name), 3)
            identifiers = self.nest.Create(name, count)
            self.cell_models[name].identifiers.extend(identifiers)
            # Check if the stitching is going OK.
            if identifiers[0] != start_id + 1:
                raise Exception("Could not match the scaffold cell identifiers to NEST identifiers! Cannot continue.")

    def connect_neurons(self, connection_models, hdf5):
        '''
            Connect the cells in NEST according to the connection model configurations
        '''
        track_models = [] # Keeps track of already added models if there'smodel=synapse_model more than 1 stitch per model
        for connection_model in connection_models.values():
            name = connection_model.name
            dataset_name = 'cells/connections/' + name
            if not dataset_name in hdf5:
                if self.scaffold.configuration.verbosity > 0:
                    print('[WARNING] Expected connection dataset "{}" not found. Skipping it.'.format(dataset_name))
                continue
            connectivity_matrix = hdf5[dataset_name]
            # Translate the id's from 0 based scaffold ID's to NEST's 1 based ID's with '+ 1'
            presynaptic_cells = np.array(connectivity_matrix[:,0] + 1, dtype=int)
            postsynaptic_cells = np.array(connectivity_matrix[:,1] + 1, dtype=int)
            if name not in track_models: # Is this the first time encountering this model?
                track_models.append(name)
                # Create the synapse model in the simulator
                self.create_synapse_model(connection_model)
                if connection_model.plastic == True:
                    # Create the volume transmitters
                    self.scaffold.report("Creating volume transmitter for "+name,3)
                    self.create_volume_transmitter(connection_model, postsynaptic_cells)
            # Set the specifications NEST allows like: 'rule', 'autapses', 'multapses'
            connection_specifications = {'rule': 'one_to_one'}
            # Get the connection parameters from the configuration
            connection_parameters = connection_model.get_connection_parameters()
            # Create the connections in NEST
            self.scaffold.report("Creating connections "+name,3)
            self.nest.Connect(presynaptic_cells, postsynaptic_cells, connection_specifications, connection_parameters)
            # Workaround for https://github.com/alberto-antonietti/CerebNEST/issues/10
            if connection_model.plastic == True:
                # Associate the presynaptic cells of each target cell to the
                # volume transmitter of that target cell
                for i,target in enumerate(postsynaptic_cells):
                    # Get connections between all presynaptic cells and target postsynaptic cell
                    connections_to_target = self.nest.GetConnections(presynaptic_cells.tolist(),[target])
                    # Associate the volume transmitter number to them
                    self.nest.SetStatus(connections_to_target ,{"vt_num": float(i)})



    def create_devices(self, devices):
        '''
            Create the configured NEST devices in the simulator
        '''
        for device_model in devices.values():
            device = self.nest.Create(device_model.device)
            self.scaffold.report("Creating device:  "+device_model.device,3)
            device_targets = device_model.get_targets()
            self.nest.SetStatus(device, device_model.parameters)
            try:
                if device_model.io == "input":
                    self.nest.Connect(device, device_targets)
                else:
                    self.nest.Connect(device_targets, device)
            except Exception as e:
                if e.errorname == 'IllegalConnection':
                    raise Exception("IllegalConnection error for '{}'".format(device_model.get_config_node())) from None

    def create_model(self, cell_model):
        '''
            Create a NEST cell model in the simulator based on a cell model configuration.
        '''
        # Use the default model unless another one is specified in the configuration.A_minus
        # Alias the nest model name under our cell model name.
        self.nest.CopyModel(cell_model.neuron_model, cell_model.name)
        # Get the synapse parameters
        params = cell_model.get_parameters()
        # Set the parameters in NEST
        self.nest.SetDefaults(cell_model.name, params)

    def create_synapse_model(self, connection_model):
        '''
            Create a NEST synapse model in the simulator based on a synapse model configuration.
        '''
        # Use the default model unless another one is specified in the configuration.
        # Alias the nest model name under our cell model name.
        self.scaffold.report("Creating synapse model '{}' for {}".format(connection_model.synapse_model, connection_model.name), 0)
        self.nest.CopyModel(connection_model.synapse_model, connection_model.name)
        # Get the synapse parameters
        params = connection_model.get_synapse_parameters()
        # Set the parameters in NEST
        self.nest.SetDefaults(connection_model.name, params)

    # This function should be simplified by providing a CreateTeacher function in the
    # CerebNEST module. See https://github.com/nest/nest-simulator/issues/1317
    # And https://github.com/alberto-antonietti/CerebNEST/issues/10
    def create_volume_transmitter(self, synapse_model, postsynaptic_cells):
        vt = self.nest.Create("volume_transmitter_alberto", len(postsynaptic_cells))
        teacher = vt[0]
        # Assign the volume transmitters to their synapse model
        self.nest.SetDefaults(synapse_model.name,{"vt": teacher})
        # Assign an ID to each volume transmitter
        for n,vti in enumerate(vt):
        	self.nest.SetStatus([vti],{"deliver_interval" : 2})            # TO CHECK
            # Waiting for Albe to clarify necessity of this parameter
        	self.nest.SetStatus([vti],{"vt_num" : n})
