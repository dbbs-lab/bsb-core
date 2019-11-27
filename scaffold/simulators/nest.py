from ..simulation import SimulatorAdapter, SimulationComponent
from ..helpers import ListEvalConfiguration
from ..exceptions import *
import os, json, weakref, numpy as np
from sklearn.neighbors import KDTree

class NestCell(SimulationComponent):

    node_name = 'simulations.?.cell_models'
    required = ['parameters']

    def boot(self):
        self.reset()
        # The cell model contains a 'parameters' attribute and many sets of
        # neuron model specific sets of parameters. Each set of neuron model
        # specific parameters can define receptor specifications.
        # Extract those if present to the designated receptor_specifications dict.
        for neuron_model in self.__dict__:
            model_parameters = self.__dict__[neuron_model]
            # Exclude the default parameters dict and transfer the receptor specifications
            if neuron_model != "parameters" and isinstance(model_parameters, dict) and "receptors" in model_parameters:
                self.receptor_specifications[neuron_model] = model_parameters["receptors"]
                del model_parameters["receptors"]

    def validate(self):
        pass

    def reset(self):
        self.nest_identifiers = []
        self.scaffold_identifiers = []
        self.scaffold_to_nest_map = {}
        self.receptor_specifications = {}

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

    def get_synapse_parameters(self, synapse_model_name):
        # Get the default synapse parameters
        return self.synapse[synapse_model_name]


    def get_connection_parameters(self):
        # Get the default synapse parameters
        params = self.connection.copy()
        # Add the receptor specifications, if required.
        if self.should_specify_receptor_type():
            # If specific receptors are specified, the weight should always be positive.
            params["weight"] = np.abs(params["weight"])
            params["receptor_type"] = self.get_receptor_type()
        params["model"] = self.suffixed(self.synapse_model)
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
        'parameters': dict,
        'stimulus': ListEvalConfiguration.cast
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
        if hasattr(self, "stimulus"):
            stimulus_name = "stimulus" if not hasattr(self.stimulus, "parameter_name") else self.stimulus.parameter_name
            self.parameters[stimulus_name] = self.stimulus.eval()

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
        'resolution': 1.0,
        'modules': ["CerebNEST"]
    }

    required = ['default_neuron_model', 'default_synapse_model',
        'duration', 'resolution', 'threads'
    ]

    def __init__(self):
        super().__init__()
        self.is_prepared = False
        self.suffix = ''
        self.multi = False
        self.has_lock = False

        def finalize_self(weak_obj):
            weak_obj().__safedel__()

        r = weakref.ref(self)
        weakref.finalize(self, finalize_self, r)

    def __safedel__(self):
        if self.has_lock:
            self.release_lock()

    def prepare(self, hdf5):
        if self.is_prepared:
            raise AdapterException("Attempting to prepare the same adapter twice. Please use `scaffold.create_adapter` for multiple adapter instances of the same simulation.")
        self.scaffold.report("Importing  NEST...", 2)
        import nest
        self.nest = nest
        self.scaffold.report("Locking NEST kernel...", 2)
        self.lock()
        self.scaffold.report("Installing  NEST modules...", 2)
        self.install_modules()
        self.scaffold.report("Initializing NEST kernel...", 2)
        self.reset_kernel()
        self.scaffold.report("Creating neurons...",2)
        self.create_neurons(self.cell_models)
        self.scaffold.report("Creating devices...",2)
        self.create_devices(self.devices)
        self.scaffold.report("Creating connections...",2)
        self.connect_neurons(self.connection_models, hdf5)
        self.is_prepared = True
        return nest

    def lock(self):
        if not self.multi:
            self.single_lock()
        else:
            self.multi_lock()
        self.has_lock = True

    def single_lock(self):
        try:
            lock_data = {"multi": False}
            self.write_lock(lock_data, mode="x")
        except FileExistsError as e:
            raise KernelLockedException("This adapter is not in multi-instance mode and another adapter is already managing the kernel.") from None

    def multi_lock(self):
        lock_data = self.read_lock()
        if lock_data is None:
            lock_data = {"multi": True, "suffixes": []}
        if not lock_data["multi"]:
            raise KernelLockedException("The kernel is locked by a single-instance adapter and cannot be managed by multiple instances.")
        if self.suffix in lock_data["suffixes"]:
            raise SuffixTakenException("The kernel is already locked by an instance with the same suffix.")
        lock_data["suffixes"].append(self.suffix)
        self.write_lock(lock_data)

    def read_lock(self):
        try:
            with open(self.get_lock_path(), "r") as lock:
                return json.loads(lock.read())
        except FileNotFoundError as e:
            return None

    def write_lock(self, lock_data, mode="w"):
        with open(self.get_lock_path(), mode) as lock:
            lock.write(json.dumps(lock_data))

    def enable_multi(self, suffix):
        self.suffix = suffix
        self.multi = True

    def release_lock(self):
        if not self.has_lock:
            raise AdapterException("Cannot unlock kernel from an adapter that has no lock on it.")
        self.has_lock = False
        lock_data = self.read_lock()
        if lock_data["multi"]:
            if len(lock_data["suffixes"]) == 1:
                self.delete_lock_file()
            else:
                lock_data["suffixes"].remove(self.suffix)
                self.write_lock(lock_data)
        else:
            self.delete_lock_file()

    def delete_lock_file(self):
        os.remove(self.get_lock_path())

    def get_lock_name(self):
        return "kernel_" + str(os.getpid()) + ".lck"

    def get_lock_path(self):
        return self.nest.__path__[0] + '/' + self.get_lock_name()

    def reset_kernel(self):
        self.nest.set_verbosity(self.verbosity)
        self.nest.ResetKernel()
        self.set_threads(self.threads)
        self.nest.SetKernelStatus({
            'resolution': self.resolution,
            'overwrite_files': True,
            'data_path': self.scaffold.output_formatter.get_simulator_output_path(self.simulator_name)
        })

    def reset(self):
        self.is_prepared = False
        self.nest.ResetKernel()

    def get_master_seed(self):
        # Use a constant reproducible master seed
        return 1989

    def set_threads(self, threads, virtual=None):
        master_seed = self.get_master_seed()
        # Update the internal reference to the amount of threads
        if virtual is None:
            virtual = threads
        # Create a range of random seeds and generators.
        random_generator_seeds = range(master_seed, master_seed + virtual)
        # Create a different range of random seeds for the kernel.
        thread_seeds = range(master_seed + virtual + 1, master_seed + 1 + 2 * virtual)
        success = True
        try:
            # Update the kernel with the new RNG and thread state.
            self.nest.SetKernelStatus({'grng_seed' : master_seed + virtual,
                                  'rng_seeds' : thread_seeds,
                                  'local_num_threads': threads,
                                  'total_num_virtual_procs': virtual,
                                 })
        except Exception as e:
            if hasattr(e, "errorname") and e.errorname[0:27] == "The resolution has been set":
                # Threads can't be updated at this point in time.
                raise NestKernelException("Updating the NEST threads or virtual processes must occur before setting the resolution.") from None
                success = False
            else:
                raise
        if success:
            self.threads = threads
            self.virtual_processes = virtual
            self.random_generators = [np.random.RandomState(seed) for seed in random_generator_seeds]

    def simulate(self, simulator):
        if not self.is_prepared:
            self.scaffold.warn("Adapter has not been prepared", SimulationWarning)
        self.scaffold.report("Simulating...", 2)
        simulator.Simulate(self.duration)
        self.scaffold.report("Simulation finished.", 2)

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
                    self.scaffold.warn("Module {} already installed".format(module), KernelWarning)
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
            nest_name = self.suffixed(name)
            if name not in track_models: # Is this the first time encountering this model?
                # Create the cell model in the simulator
                self.scaffold.report("Creating " + nest_name + "...", 3)
                self.create_model(cell_models[name])
                track_models.append(name)
            # Create the same amount of cells that were placed in this stitch.
            self.scaffold.report("Creating {} {}...".format(count, nest_name), 3)
            identifiers = self.nest.Create(nest_name, count)
            self.cell_models[name].identifiers.extend(identifiers)

    def connect_neurons(self, connection_models, hdf5):
        '''
            Connect the cells in NEST according to the connection model configurations
        '''
        track_models = [] # Keeps track of already added models if there'smodel=synapse_model more than 1 stitch per model
        for connection_model in connection_models.values():
            name = connection_model.name
            nest_name = self.suffixed(name)
            dataset_name = 'cells/connections/' + name
            if not dataset_name in hdf5:
                self.scaffold.warn('Expected connection dataset "{}" not found. Skipping it.'.format(dataset_name), ConnectivityWarning)
                continue
            connectivity_matrix = hdf5[dataset_name]
            # Get the NEST identifiers for the connections made in the connectivity matrix
            presynaptic_cells = self.get_nest_ids(np.array(connectivity_matrix[:,0], dtype=int))
            postsynaptic_cells = self.get_nest_ids(np.array(connectivity_matrix[:,1], dtype=int))
            if name not in track_models: # Is this the first time encountering this model?
                track_models.append(name)
                # Create the synapse model in the simulator
                self.create_synapse_model(connection_model)
                # Create the volume transmitter if the connection is plastic with heterosynaptic plasticity
                if connection_model.plastic == True and connection_model.hetero == True:
                    # Create the volume transmitters
                    self.scaffold.report("Creating volume transmitter for " + nest_name,3)
                    self.create_volume_transmitter(connection_model, postsynaptic_cells)
            # Set the specifications NEST allows like: 'rule', 'autapses', 'multapses'
            connection_specifications = {'rule': 'one_to_one'}
            # Get the connection parameters from the configuration
            connection_parameters = connection_model.get_connection_parameters()
            # Create the connections in NEST
            self.scaffold.report("Creating connections " + nest_name,3)
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
            # Execute SetStatus and catch DictError
            self.execute_command(self.nest.SetStatus, device, device_model.parameters,
                exceptions={
                'DictError': {
                    'from': None,
                    'exception': catch_dict_error("Could not create {} device '{}': ".format(
                        device_model.device, device_model.name
                    ))
                }
            })
            device_targets = device_model.get_targets()
            self.scaffold.report("Connecting to {} device targets.".format(len(device_targets)), 3)
            try:
                if device_model.io == "input":
                    self.nest.Connect(device, device_targets)
                elif device_model.io == "output":
                    self.nest.Connect(device_targets, device)
                else:
                    pass                # Weight recorder device is not connected to any node; just linked to a connection
            except Exception as e:
                if e.errorname == 'IllegalConnection':
                    raise Exception("IllegalConnection error for '{}'".format(device_model.get_config_node())) from None
                else:
                    raise

    def create_model(self, cell_model):
        '''
            Create a NEST cell model in the simulator based on a cell model configuration.
        '''
        # Use the default model unless another one is specified in the configuration.A_minus
        # Alias the nest model name under our cell model name.
        nest_name = self.suffixed(cell_model.name)
        self.nest.CopyModel(cell_model.neuron_model, nest_name)
        # Get the synapse parameters
        params = cell_model.get_parameters()
        # Set the parameters in NEST
        self.nest.SetDefaults(nest_name, params)

    def create_synapse_model(self, connection_model):
        '''
            Create a NEST synapse model in the simulator based on a synapse model configuration.
        '''
        nest_name = self.suffixed(connection_model.name)
        # Use the default model unless another one is specified in the configuration.
        # Alias the nest model name under our cell model name.
        self.scaffold.report("Creating synapse model '{}' for {}".format(connection_model.synapse_model, nest_name), 0)
        self.nest.CopyModel(connection_model.synapse_model, nest_name)
        # Get the synapse parameters
        params = connection_model.get_synapse_parameters(connection_model.synapse_model)
        # Set the parameters in NEST
        self.nest.SetDefaults(nest_name, params)

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

    def execute_command(self, command, *args, exceptions={}):
        try:
            command(*args)
        except Exception as e:
            if not hasattr(e, "errorname"):
                raise
            if e.errorname in exceptions:
                handler = exceptions[e.errorname ]
                if "from" in handler:
                    raise handler["exception"](e) from handler["from"]
                else:
                    raise handler["exception"]
            else:
                raise

    def suffixed(self, str):
        if self.suffix == '':
            return str
        return str + '_' + self.suffix

def catch_dict_error(message):
    def handler(e):
        attributes = list(map(lambda x: x.strip(), e.errormessage.split(":")[-1].split(",")))
        return NestModelException(message + "Unknown attributes {}".format("'" + "', '".join(attributes) + "'"))

    return handler
