from ..simulation import (
    SimulatorAdapter,
    SimulationComponent,
    SimulationCell,
    TargetsNeurons,
)
from ..models import ConnectivitySet
from ..helpers import ListEvalConfiguration, listify_input
from ..reporting import report, warn
from ..exceptions import *
import time, os, json, weakref, numpy as np
from itertools import chain
from copy import deepcopy
from sklearn.neighbors import KDTree
from ..simulation import SimulationRecorder, SimulationResult
import warnings
import h5py
import time

try:
    from functools import cached_property, cache
except:
    from functools import lru_cache

    cache = lru_cache(None)

    def cached_property(f):
        return property(cache(f))


try:
    import mpi4py
    import mpi4py.MPI

    _MPI_processes = mpi4py.MPI.COMM_WORLD.Get_size()
    _MPI_rank = mpi4py.MPI.COMM_WORLD.Get_rank()
except ImportError as e:
    warn(f"Could not import `mpi4py.MPI`: {e}")
    _MPI_processes = 1
    _MPI_rank = 0

_LOCK_ATTRIBUTE = "_dbbs_scaffold_lock"
_HOT_MODULE_ATTRIBUTE = "_dbbs_scaffold_hot_modules"


class MapsScaffoldIdentifiers:
    def reset_identifiers(self):
        self.nest_identifiers = []
        self.scaffold_identifiers = []
        self.scaffold_to_nest_map = {}

    def _build_identifier_map(self):
        self.scaffold_to_nest_map = dict(
            zip(self.scaffold_identifiers, self.nest_identifiers)
        )

    def get_nest_ids(self, ids):
        return [self.scaffold_to_nest_map[id] for id in ids]


class NestCell(SimulationCell, MapsScaffoldIdentifiers):

    node_name = "simulations.?.cell_models"

    def boot(self):
        super().boot()
        self.receptor_specifications = {}
        self.reset()
        if self.relay:
            # If a cell type is marked as a relay then the cell model should be a
            # parameterless "parrot_neuron" model if no specifics are provided.
            #
            # Set the default relay model to "parrot_neuron"
            if not hasattr(self, "neuron_model"):
                self.neuron_model = "parrot_neuron"
            # Set the default parameter dict to empty
            if not hasattr(self, "parameters"):
                self.parameters = {}
            # Set the default relay model parameter dict to empty
            if not hasattr(self, self.neuron_model):
                self.__dict__[self.neuron_model] = {}

        # The cell model contains a 'parameters' attribute and many sets of
        # neuron model specific sets of parameters. Each set of neuron model
        # specific parameters can define receptor specifications.
        # Extract those if present to the designated receptor_specifications dict.
        for neuron_model in self.__dict__:
            model_parameters = self.__dict__[neuron_model]
            # Iterate over the model specific parameter dicts with receptor
            # specifications, excluding the default parameter dict.
            if (
                neuron_model != "parameters"
                and isinstance(model_parameters, dict)
                and "receptors" in model_parameters
            ):
                # Transfer the receptor specifications
                self.receptor_specifications[neuron_model] = model_parameters["receptors"]
                del model_parameters["receptors"]

    def validate(self):
        if not self.relay and not hasattr(self, "parameters"):
            raise AttributeMissingError(
                "Required attribute 'parameters' missing from '{}'".format(
                    self.get_config_node()
                )
            )

    def reset(self):
        self.reset_identifiers()

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        # Raise an exception if the requested model is not configured.
        if not hasattr(self, self.neuron_model):
            raise ConfigurationError(
                "Missing parameters for '{}' model in '{}'".format(
                    self.neuron_model, self.name
                )
            )
        # Merge in the model specific parameters
        params.update(self.__dict__[self.neuron_model])
        return params

    def get_receptor_specifications(self):
        if self.neuron_model in self.receptor_specifications:
            return self.receptor_specifications[self.neuron_model]
        else:
            return {}


class NestConnection(SimulationComponent):
    node_name = "simulations.?.connection_models"

    casts = {"synapse": dict, "connection": dict}

    required = ["synapse", "connection"]

    defaults = {
        "plastic": False,
        "hetero": None,
        "teaching": None,
        "is_teaching": False,
    }

    def validate(self):
        if "weight" not in self.connection:
            raise ConfigurationError(
                "Missing 'weight' in the connection parameters of "
                + self.node_name
                + "."
                + self.name
            )
        if self.plastic:
            # Set plasticity synapse dict defaults
            synapse_defaults = {
                "A_minus": 0.0,
                "A_plus": 0.0,
                "Wmin": 0.0,
                "Wmax": 4000.0,
            }
            for key, value in synapse_defaults.items():
                if key not in self.synapse:
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
            # We try to sanitize user data as best we can. If the given weight is a distr
            # (given as a dict) we try to sanitize the `mu` value, if present.
            if type(params["weight"]) is dict:
                if "mu" in params["weight"].keys():
                    params["weight"]["mu"] = np.abs(params["weight"]["mu"])
            else:
                params["weight"] = np.abs(params["weight"])
            if "Wmax" in params:
                params["Wmax"] = np.abs(params["Wmax"])
            if "Wmin" in params:
                params["Wmin"] = np.abs(params["Wmin"])
            params["receptor_type"] = self.get_receptor_type()
        params["model"] = self.adapter.suffixed(self.name)
        return params

    def _get_cell_types(self, key="from"):
        meta = self.scaffold.output_formatter.get_connectivity_set_meta(self.name)
        if key + "_cell_types" in meta:
            cell_types = set()
            for name in meta[key + "_cell_types"]:
                cell_types.add(self.scaffold.get_cell_type(name))
            return list(cell_types)
        connection_types = (
            self.scaffold.output_formatter.get_connectivity_set_connection_types(
                self.name
            )
        )
        cell_types = set()
        for connection_type in connection_types:
            cell_types |= set(connection_type.__dict__[key + "_cell_types"])
        return list(cell_types)

    def get_cell_types(self):
        return self._get_cell_types(key="from"), self._get_cell_types(key="to")

    def should_specify_receptor_type(self):
        _, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisiting of more than 1 cell type is currently undefined behaviour."
            )
        to_cell_type = to_cell_types[0]
        to_cell_model = self.adapter.cell_models[to_cell_type.name]
        return to_cell_model.neuron_model in to_cell_model.receptor_specifications

    def get_receptor_type(self):
        from_cell_types, to_cell_types = self.get_cell_types()
        if len(to_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisiting of more than 1 target cell type is currently undefined behaviour."
            )
        if len(from_cell_types) > 1:
            raise NotImplementedError(
                "Specifying receptor types of connections consisting of more than 1 origin cell type is currently undefined behaviour."
            )
        to_cell_type = to_cell_types[0]
        from_cell_type = from_cell_types[0]
        to_cell_model = self.adapter.cell_models[to_cell_type.name]
        if from_cell_type.name in self.adapter.cell_models.keys():
            from_cell_model = self.adapter.cell_models[from_cell_type.name]
        else:  # For neurons receiving from entities
            from_cell_model = self.adapter.entities[from_cell_type.name]
        receptors = to_cell_model.get_receptor_specifications()
        if from_cell_model.name not in receptors:
            raise ReceptorSpecificationError(
                "Missing receptor specification for cell model '{}' in '{}' while attempting to connect a '{}' to it during '{}'".format(
                    to_cell_model.name, self.node_name, from_cell_model.name, self.name
                )
            )
        return receptors[from_cell_model.name]


class NestDevice(TargetsNeurons, SimulationComponent):
    node_name = "simulations.?.devices"

    casts = {
        "radius": float,
        "origin": [float],
        "parameters": dict,
        "stimulus": ListEvalConfiguration.cast,
    }

    defaults = {"connection": {"rule": "all_to_all"}, "synapse": None}

    required = ["targetting", "device", "io", "parameters"]

    def validate(self):
        if self.io not in ("input", "output"):
            raise ConfigurationError(
                "Attribute io needs to be either 'input' or 'output' in {}".format(
                    self.node_name
                )
            )
        if hasattr(self, "stimulus"):
            stimulus_name = (
                "stimulus"
                if not hasattr(self.stimulus, "parameter_name")
                else self.stimulus.parameter_name
            )
            self.parameters[stimulus_name] = self.stimulus.eval()

    def boot(self):
        super().boot()
        self.protocol = get_device_protocol(self)

    def get_nest_targets(self):
        """
        Return the targets of the stimulation to pass into the nest.Connect call.
        """
        targets = np.array(self.get_targets(), dtype=int)
        return self.adapter.get_nest_ids(targets)


class NestEntity(NestDevice, MapsScaffoldIdentifiers):
    node_name = "simulations.?.entities"

    def boot(self):
        super().boot()
        self.reset_identifiers()


class NestAdapter(SimulatorAdapter):
    """
    Interface between the scaffold model and the NEST simulator.
    """

    simulator_name = "nest"

    configuration_classes = {
        "cell_models": NestCell,
        "connection_models": NestConnection,
        "devices": NestDevice,
        "entities": NestEntity,
    }

    casts = {"threads": int, "modules": list}

    defaults = {
        "default_synapse_model": "static_synapse",
        "default_neuron_model": "iaf_cond_alpha",
        "verbosity": "M_ERROR",
        "threads": 1,
        "resolution": 1.0,
        "modules": [],
    }

    required = [
        "default_neuron_model",
        "default_synapse_model",
        "duration",
        "resolution",
        "threads",
    ]

    @cached_property
    def nest(self):
        report("Importing  NEST...", level=2)
        import nest

        self._nest = nest
        setattr(nest, _HOT_MODULE_ATTRIBUTE, set())
        return self._nest

    def __init__(self):
        super().__init__()
        self.result = SimulationResult()
        self.is_prepared = False
        self.suffix = ""
        self.multi = False
        self.has_lock = False
        self.global_identifier_map = {}
        self.simulation_id = _randint()

    def prepare(self):
        if self.is_prepared:
            raise AdapterError(
                "Attempting to prepare the same adapter twice. Please use `bsb.create_adapter` for multiple adapter instances of the same simulation."
            )
        report("Locking NEST kernel...", level=2)
        self.lock()
        report("Installing  NEST modules...", level=2)
        self.install_modules()
        if self.in_full_control():
            report("Initializing NEST kernel...", level=2)
            self.reset_kernel()
        report("Creating neurons...", level=2)
        self.create_neurons()
        report("Creating entities...", level=2)
        self.create_entities()
        report("Building identifier map...", level=2)
        self._build_identifier_map()
        report("Creating devices...", level=2)
        self.create_devices()
        report("Creating connections...", level=2)
        self.connect_neurons()
        self.is_prepared = True
        return self.nest

    def get_rank(self):
        return _MPI_rank

    def get_size(self):
        return _MPI_processes

    def broadcast(self, data, root=0):
        return mpi4py.MPI.COMM_WORLD.bcast(data, root)

    def in_full_control(self):
        if not self.has_lock or not self.read_lock():
            raise AdapterError(
                "Can't check if we're in full control of the kernel: we have no lock on the kernel."
            )
        return not self.multi or len(self.read_lock()["suffixes"]) == 1

    def lock(self):
        if not self.multi:
            self.single_lock()
        else:
            self.multi_lock()
        self.has_lock = True

    def single_lock(self):
        if hasattr(self.nest, _LOCK_ATTRIBUTE):
            raise KernelLockedError(
                "This adapter is not in multi-instance mode and another adapter is already managing the kernel."
            )
        else:
            lock_data = {"multi": False}
            self.write_lock(lock_data)

    def multi_lock(self):
        lock_data = self.read_lock()
        if lock_data is None:
            lock_data = {"multi": True, "suffixes": []}
        if not lock_data["multi"]:
            raise KernelLockedError(
                "The kernel is locked by a single-instance adapter and cannot be managed by multiple instances."
            )
        if self.suffix in lock_data["suffixes"]:
            raise SuffixTakenError(
                "The kernel is already locked by an instance with the same suffix."
            )
        lock_data["suffixes"].append(self.suffix)
        self.write_lock(lock_data)

    def read_lock(self):
        if hasattr(self.nest, _LOCK_ATTRIBUTE):
            return getattr(self.nest, _LOCK_ATTRIBUTE)
        else:
            return None

    def write_lock(self, lock_data):
        setattr(self.nest, _LOCK_ATTRIBUTE, lock_data)

    def enable_multi(self, suffix):
        self.suffix = suffix
        self.multi = True

    def release_lock(self):
        if not self.has_lock:
            raise AdapterError(
                "Cannot unlock kernel from an adapter that has no lock on it."
            )
        self.has_lock = False
        lock_data = self.read_lock()
        if lock_data["multi"]:
            if len(lock_data["suffixes"]) == 1:
                self.delete_lock()
            else:
                lock_data["suffixes"].remove(self.suffix)
                self.write_lock(lock_data)
        else:
            self.delete_lock()

    def delete_lock(self):
        try:
            delattr(self.nest, _LOCK_ATTRIBUTE)
        except AttributeError:
            pass

    def get_rank(self):
        return mpi4py.MPI.COMM_WORLD.Get_rank()

    def reset_kernel(self):
        self.nest.set_verbosity(self.verbosity)
        self.nest.ResetKernel()
        # Reset which modules we should consider explicitly loaded by the user
        # to appropriately warn them when they load them twice.
        setattr(self.nest, _HOT_MODULE_ATTRIBUTE, set())
        self.reset_processes(self.threads)
        self.nest.SetKernelStatus(
            {
                "resolution": self.resolution,
                "overwrite_files": True,
                "data_path": self.scaffold.output_formatter.get_simulator_output_path(
                    self.simulator_name
                ),
            }
        )

    def reset(self):
        self.is_prepared = False
        if hasattr(self, "nest"):
            self.reset_kernel()
        self.global_identifier_map = {}
        for cell_model in self.cell_models.values():
            cell_model.reset()
        if self.has_lock:
            self.release_lock()

    def get_master_seed(self, fixed_seed=None):
        if not hasattr(self, "_master_seed"):
            if fixed_seed is None:
                # Use time as random seed
                if mpi4py.MPI.COMM_WORLD.rank == 0:
                    fixed_seed = int(time.time())
                else:
                    fixed_seed = None
                self._master_seed = mpi4py.MPI.COMM_WORLD.bcast(fixed_seed, root=0)
            else:
                self._master_seed = fixed_seed
        return self._master_seed

    def reset_processes(self, threads):
        master_seed = self.get_master_seed()
        total_num = _MPI_processes * threads
        # Create a range of random seeds and generators.
        random_generator_seeds = range(master_seed, master_seed + total_num)
        # Create a different range of random seeds for the kernel.
        thread_seeds = range(master_seed + 1 + total_num, master_seed + 1 + 2 * total_num)
        success = True
        try:
            # Update the kernel with the new RNG and thread state.
            self.nest.SetKernelStatus(
                {
                    "grng_seed": master_seed + total_num,
                    "rng_seeds": thread_seeds,
                    "local_num_threads": threads,
                    "total_num_virtual_procs": total_num,
                }
            )
        except Exception as e:
            if (
                hasattr(e, "errorname")
                and e.errorname[0:27] == "The resolution has been set"
            ):
                # Threads can't be updated at this point in time.
                success = False
                raise NestKernelError(
                    "Updating the NEST threads or virtual processes must occur before setting the resolution."
                ) from None
            else:
                raise
        if success:
            self.threads_per_node = threads
            self.virtual_processes = total_num
            self.random_generators = [
                np.random.RandomState(seed) for seed in random_generator_seeds
            ]

    def simulate(self, simulator):
        if not self.is_prepared:
            warn("Adapter has not been prepared", SimulationWarning)
        report("Simulating...", level=2)
        tick = time.time()
        self.start_progress(self.duration)
        with simulator.RunManager():
            for oi, i in self.step_progress(self.duration, step=1):
                simulator.Run(i - oi)
                self.progress(i)
        report(f"Simulation done. {time.time() - tick:.2f}s elapsed.", level=2)
        if self.has_lock:
            self.release_lock()

    def step(self, simulator, dt):
        if not self.is_prepared:
            warn("Adapter has not been prepared", SimulationWarning)
        report("Simulating...", level=2)
        tick = time.time()
        self.start_progress(self.duration)
        with simulator.RunManager():
            for oi, i in self.step_progress(self.duration, step=dt):
                simulator.Run(i - oi)
                yield self.progress(i)

        report(f"Simulation done. {time.time() - tick:.2f}s elapsed.", level=2)
        if self.has_lock:
            self.release_lock()

    def collect_output(self, simulator):
        report("Collecting output...", level=2)
        tick = time.time()
        try:
            import mpi4py

            rank = mpi4py.MPI.COMM_WORLD.rank
        except Exception as e:
            print(str(e))
            rank = 0

        timestamp = str(time.time()).split(".")[0] + str(_randint())
        result_path = "results_" + self.name + "_" + timestamp + ".hdf5"
        if rank == 0:
            with h5py.File(result_path, "a") as f:
                f.attrs["configuration_string"] = self.scaffold.configuration._raw
                for path, data, meta in self.result.safe_collect():
                    try:
                        path = "/".join(path)
                        if path in f:
                            data = np.vstack((f[path][()], data))
                            del f[path]
                        d = f.create_dataset(path, data=data)
                        for k, v in meta.items():
                            d.attrs[k] = v
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        if not isinstance(data, np.ndarray):
                            warn(
                                "Recorder {} numpy.ndarray expected, got {}".format(
                                    path, type(data)
                                )
                            )
                        else:
                            warn(
                                "Recorder {} processing errored out: {}".format(
                                    path, "{} {}".format(data.dtype, data.shape)
                                )
                            )
        mpi4py.MPI.COMM_WORLD.bcast(result_path, root=0)
        report(
            f"Output collected in '{result_path}'. "
            + f"{time.time() - tick:.2f}s elapsed.",
            level=2,
        )
        return result_path

    def validate(self):
        for cell_model in self.cell_models.values():
            cell_model.neuron_model = (
                cell_model.neuron_model
                if hasattr(cell_model, "neuron_model")
                else self.default_neuron_model
            )
        for connection_model in self.connection_models.values():
            connection_model.synapse_model = (
                connection_model.synapse_model
                if hasattr(connection_model, "synapse_model")
                else self.default_synapse_model
            )
            connection_model.plastic = (
                connection_model.plastic
                if hasattr(connection_model, "plastic")
                else connection_model.defaults["plastic"]
            )
            connection_model.hetero = (
                connection_model.hetero
                if hasattr(connection_model, "hetero")
                else connection_model.defaults["hetero"]
            )
            if connection_model.plastic and connection_model.hetero:
                if not hasattr(connection_model, "teaching"):
                    raise ConfigurationError(
                        "Required attribute 'teaching' is missing for heteroplastic connection '{}'".format(
                            connection_model.get_config_node()
                        )
                    )
                if connection_model.teaching not in self.connection_models:
                    raise ConfigurationError(
                        "Teaching connection '{}' does not exist".format(
                            connection_model.teaching
                        )
                    )
                # Set the is_teaching parameter of teaching connection to true
                teaching_connection = self.connection_models[connection_model.teaching]
                teaching_connection.is_teaching = True
                teaching_connection.add_after(connection_model.name)

    def install_modules(self):
        for module in self.modules:
            hot = getattr(self.nest, _HOT_MODULE_ATTRIBUTE)
            try:
                self.nest.Install(module)
                hot.add(module)
            except Exception as e:
                if e.errorname == "DynamicModuleManagementError":
                    if "loaded already" in e.message:
                        # Modules stay loaded in between `ResetKernel` calls. We
                        # assume that there's nothing to warn the user about if
                        # the adapter installs the modules each
                        # `reset`/`prepare` cycle.
                        if module in hot:
                            warn(f"Already installed '{module}'.", KernelWarning)
                    elif "file not found" in e.message:
                        raise NestModuleError(
                            "Module {} not found".format(module)
                        ) from None
                    else:
                        raise
                else:
                    raise

    def _build_identifier_map(self):
        # Iterate over all simulation components that contain representations
        # of scaffold components with an ID to create a map of all scaffold ID's
        # to all NEST ID's this adapter manages
        for mapping_type in chain(self.entities.values(), self.cell_models.values()):
            # "Freeze" the type's identifiers into a map
            mapping_type._build_identifier_map()
            # Add the type's map to the global map
            self.global_identifier_map.update(mapping_type.scaffold_to_nest_map)

    def get_nest_ids(self, ids):
        return [self.global_identifier_map[id] for id in ids]

    def get_scaffold_ids(self, ids):
        scaffold_map = {v: k for k, v in self.global_identifier_map.items()}
        return [scaffold_map[id] for id in ids]

    def create_neurons(self):
        """
        Create a population of nodes in the NEST simulator based on the cell model
        configurations.
        """
        for cell_model in self.cell_models.values():
            # Get the cell type's placement information
            ps = self.scaffold.get_placement_set(cell_model.name)
            nest_name = self.suffixed(cell_model.name)
            # Create the population's model
            self.create_model(cell_model)
            scaffold_identifiers = ps.identifiers
            report(
                "Creating {} {}...".format(len(scaffold_identifiers), nest_name), level=3
            )
            nest_identifiers = self.nest.Create(nest_name, len(scaffold_identifiers))
            cell_model.scaffold_identifiers.extend(scaffold_identifiers)
            cell_model.nest_identifiers.extend(nest_identifiers)

    def create_entities(self):
        # Create entities
        for entity_type in self.entities.values():
            name = entity_type.name
            nest_name = self.suffixed(name)
            count = self.scaffold.statistics.cells_placed[entity_type.name]
            # Create the cell model in the simulator
            report("Creating " + nest_name + "...", level=3)
            entity_nodes = list(self.nest.Create(entity_type.device, count))
            report("Creating {} {}...".format(count, nest_name), level=3)
            if hasattr(entity_type, "parameters"):
                # Execute SetStatus and catch DictError
                self.execute_command(
                    self.nest.SetStatus,
                    entity_nodes,
                    entity_type.parameters,
                    exceptions={
                        "DictError": {
                            "from": None,
                            "exception": catch_dict_error(
                                "Could not create {} device '{}': ".format(
                                    entity_type.device, entity_type.name
                                )
                            ),
                        }
                    },
                )
            entity_type.scaffold_identifiers = self.scaffold.get_entities_by_type(
                entity_type.name
            )
            entity_type.nest_identifiers = entity_nodes

    def connect_neurons(self):
        """
        Connect the cells in NEST according to the connection model configurations
        """
        order = NestConnection.resolve_order(self.connection_models)

        for connection_model in order:
            name = connection_model.name
            nest_name = self.suffixed(name)
            cs = ConnectivitySet(self.scaffold.output_formatter, name)
            if not cs.exists():
                warn(
                    'Expected connection dataset "{}" not found. Skipping it.'.format(
                        name
                    ),
                    ConnectivityWarning,
                )
                continue
            # Get the NEST identifiers for the connections made in the connectivity matrix
            try:
                presynaptic_sources = np.array(
                    self.get_nest_ids(np.array(cs.from_identifiers, dtype=int))
                )
            except KeyError as e:
                raise UnknownGIDError(
                    f"Unknown GID {e.args[0]} in presynaptic `{name}` data."
                ) from None
            try:
                postsynaptic_targets = np.array(
                    self.get_nest_ids(np.array(cs.to_identifiers, dtype=int))
                )
            except KeyError as e:
                raise UnknownGIDError(
                    f"Unknown GID {e.args[0]} in postsynaptic `{name}` data."
                ) from None
            if not len(presynaptic_sources) or not len(postsynaptic_targets):
                warn("No connections for " + name)
                continue
            # Accessing the postsynaptic type to be associated to the volume transmitter of the synapse
            postsynaptic_type = cs.connection_types[0].to_cell_types[0]
            postsynaptic_cells = np.unique(postsynaptic_targets)

            # Create the synapse model in the simulator
            self.create_synapse_model(connection_model)
            # Set the specifications NEST allows like: 'rule', 'autapses', 'multapses'
            connection_specifications = {"rule": "one_to_one"}
            if hasattr(self, "weight_recorder"):
                wr_conf = self.weight_recorder
                wr = nest.Create("weight_recorder")
                nest.SetStatus(wr, wr_conf)
                connection_specifications["weight_recorder"] = wr
            # Get the connection parameters from the configuration
            connection_parameters = connection_model.get_connection_parameters()
            report("Creating connections " + nest_name, level=3)
            # Create the connections in NEST
            if not (connection_model.plastic and connection_model.hetero):
                receptor_cfg = connection_parameters.get("receptor_type", [])
                # Repeat connections per receptor type
                receptor_types = listify_input(receptor_cfg)
                if not len(receptor_types):
                    # If no receptor types are specified, go over the connection loop
                    # once, without setting any receptor type in the conn params.
                    receptor_types.append(None)
                for receptor_type in receptor_types:
                    single_connection_parameters = deepcopy(connection_parameters)
                    if receptor_type is not None:
                        single_connection_parameters["receptor_type"] = receptor_type
                    self.execute_command(
                        self.nest.Connect,
                        presynaptic_sources,
                        postsynaptic_targets,
                        connection_specifications,
                        single_connection_parameters,
                        exceptions={
                            "IncompatibleReceptorType": {
                                "from": None,
                                "exception": catch_receptor_error(
                                    "Invalid receptor specifications in {}: ".format(name)
                                ),
                            }
                        },
                    )
            else:
                # Create the volume transmitter if the connection is plastic with heterosynaptic plasticity
                report("Creating volume transmitter for " + name, level=3)
                volume_transmitters = self.create_volume_transmitter(
                    connection_model, postsynaptic_cells
                )
                postsynaptic_type._vt_id = volume_transmitters

                # Each post synaptic cell has to set its own vt_num for its synapses
                for vt_num, post_cell in enumerate(postsynaptic_cells):
                    connection_parameters = connection_model.get_connection_parameters()
                    connection_parameters["vt_num"] = float(vt_num)
                    indexes = np.where(postsynaptic_targets == post_cell)[0]
                    pre_neurons = presynaptic_sources[indexes]
                    post_neurons = postsynaptic_targets[indexes]

                    self.execute_command(
                        self.nest.Connect,
                        pre_neurons,
                        post_neurons,
                        connection_specifications,
                        connection_parameters,
                        exceptions={
                            "IncompatibleReceptorType": {
                                "from": None,
                                "exception": catch_receptor_error(
                                    "Invalid receptor specifications in {}: ".format(name)
                                ),
                            }
                        },
                    )

            if connection_model.is_teaching:
                # We need to map the ID of the postsynaptic_target to its relative volume_transmitter
                min_ID_postsynaptic = np.min(postsynaptic_targets)
                min_ID_volume_transmitter = np.min(postsynaptic_type._vt_id)
                delta_ID = min_ID_volume_transmitter - min_ID_postsynaptic
                postsynaptic_volume_transmitters = postsynaptic_targets + delta_ID
                self.nest.Connect(
                    presynaptic_sources,
                    postsynaptic_volume_transmitters,
                    connection_specifications,
                    {"model": "static_synapse", "weight": 1.0, "delay": 1.0},
                )

    def create_devices(self):
        """
        Create the configured NEST devices in the simulator
        """
        for device_model in self.devices.values():
            device_model.initialise_targets()
            device_model.protocol.before_create()
            device = self.nest.Create(device_model.device)
            report("Creating device:  " + device_model.device, level=3)
            # Execute SetStatus and catch DictError
            self.execute_command(
                self.nest.SetStatus,
                device,
                device_model.parameters,
                exceptions={
                    "DictError": {
                        "from": None,
                        "exception": catch_dict_error(
                            "Could not create {} device '{}': ".format(
                                device_model.device, device_model.name
                            )
                        ),
                    }
                },
            )
            device_model.protocol.after_create(device)
            # Execute targetting mechanism to fetch target NEST ID's
            device_targets = device_model.get_nest_targets()
            report(
                "Connecting to {} device targets.".format(len(device_targets)), level=3
            )
            # Collect the NEST Connect parameters
            if device_model.io == "input":
                # Connect device to nodes
                connect_params = [device, device_targets]
            elif device_model.io == "output":
                # Connect nodes to device
                connect_params = [device_targets, device]
            elif device_model.io == "none":
                # Weight recorder device is not connected to any node; just linked to a connection
                return
            else:
                raise ConfigurationError(
                    "Unknown device type '{}' for {}".format(
                        device_model.io, device_model.name
                    )
                )
            connect_params.append(device_model.connection)
            connect_params.append(device_model.synapse)
            # Send the Connect command to NEST and catch IllegalConnection errors.
            self.execute_command(
                self.nest.Connect,
                *connect_params,
                exceptions={
                    "IllegalConnection": {
                        "from": None,
                        "exception": catch_connection_error(
                            device_model.get_config_node()
                        ),
                    }
                },
            )

    def create_model(self, cell_model):
        """
        Create a NEST cell model in the simulator based on a cell model configuration.
        """
        # Use the default model unless another one is specified in the configuration.A_minus
        # Alias the nest model name under our cell model name.
        nest_name = self.suffixed(cell_model.name)
        self.nest.CopyModel(cell_model.neuron_model, nest_name)
        # Get the synapse parameters
        params = cell_model.get_parameters()
        # Set the parameters in NEST
        self.nest.SetDefaults(nest_name, params)

    def create_synapse_model(self, connection_model):
        """
        Create a NEST synapse model in the simulator based on a synapse model configuration.
        """
        nest_name = self.suffixed(connection_model.name)
        # Use the default model unless another one is specified in the configuration.
        # Alias the nest model name under our cell model name.
        report(
            "Copying synapse model '{}' to {}".format(
                connection_model.synapse_model, nest_name
            ),
            level=3,
        )
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
        nest_name = self.suffixed(synapse_model.name)
        self.nest.SetDefaults(nest_name, {"vt": teacher})
        # Assign an ID to each volume transmitter
        for n, vti in enumerate(vt):
            self.nest.SetStatus([vti], {"vt_num": n})
        return vt

    def execute_command(self, command, *args, exceptions={}):
        try:
            command(*args)
        except Exception as e:
            if not hasattr(e, "errorname"):
                raise
            if e.errorname in exceptions:
                handler = exceptions[e.errorname]
                if "from" in handler:
                    raise handler["exception"](e) from handler["from"]
                else:
                    raise handler["exception"]
            else:
                raise

    def suffixed(self, str):
        if self.suffix == "":
            return str
        return str + "_" + self.suffix


def catch_dict_error(message):
    def handler(e):
        attributes = list(
            map(lambda x: x.strip(), e.errormessage.split(":")[-1].split(","))
        )
        return NestModelError(
            message + "Unknown attributes {}".format("'" + "', '".join(attributes) + "'")
        )

    return handler


def catch_receptor_error(message):
    def handler(e):
        return NestModelError(message + e.errormessage.split(":")[-1].strip())

    return handler


def catch_connection_error(source):
    def handler(e):
        return NestModelError(
            "Illegal connections for '{}'".format(source) + ": " + e.errormessage
        )

    return handler


class SpikeRecorder(SimulationRecorder):
    def __init__(self, device_model):
        self.device_model = device_model

    def get_path(self):
        return ("recorders", "soma_spikes", self.device_model.name)

    def get_data(self):
        from glob import glob

        files = glob("*" + self.device_model.parameters["label"] + "*.gdf")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spikes = np.zeros((0, 2), dtype=float)
            for file in files:
                file_spikes = np.loadtxt(file)
                # if len(file_spikes):
                if len(file_spikes.shape) > 1:
                    scaffold_ids = np.array(
                        self.device_model.adapter.get_scaffold_ids(file_spikes[:, 0])
                    )
                    self.cell_types = list(
                        set(
                            self.device_model.adapter.scaffold.get_gid_types(scaffold_ids)
                        )
                    )
                    times = file_spikes[:, 1]
                    scaffold_spikes = np.column_stack((scaffold_ids, times))
                    spikes = np.concatenate((spikes, scaffold_spikes))
                os.remove(file)
        return spikes

    def get_meta(self):
        if hasattr(self.device_model, "cell_types"):
            self.cell_types = [
                self.device_model.adapter.scaffold.get_cell_type(n)
                for n in self.device_model.cell_types
            ]
        else:
            self.cell_types = list(
                set(
                    self.device_model.adapter.scaffold.get_gid_types(
                        self.device_model.get_nest_targets()
                    )
                )
            )
        return {
            "name": self.device_model.name,
            "label": self.cell_types[0].name,
            "cell_types": [ct.name for ct in self.cell_types],
            "color": self.cell_types[0].plotting.color,
            "parameters": json.dumps(self.device_model.parameters),
        }


def _randint():
    return np.random.randint(np.iinfo(int).max)


class DeviceProtocol:
    def __init__(self, device):
        self.device = device

    def before_create(self):
        pass

    def after_create(self, id):
        pass


class SpikeDetectorProtocol(DeviceProtocol):
    def before_create(self):
        if "label" not in self.device.parameters:
            raise ConfigurationError(
                "Required `label` missing in spike detector '{}' parameters.".format(
                    self.device.name
                )
            )
        device_tag = str(_randint())
        device_tag = mpi4py.MPI.COMM_WORLD.bcast(device_tag, root=0)
        if not hasattr(self.device, "_orig_label"):
            self.device._orig_label = self.device.parameters["label"]
        self.device.parameters["label"] = self.device._orig_label + device_tag
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            self.device.adapter.result.add(SpikeRecorder(self.device))


def get_device_protocol(device):
    if device.device in _device_protocols:
        return _device_protocols[device.device](device)
    return DeviceProtocol(device)


_device_protocols = {"spike_detector": SpikeDetectorProtocol}
