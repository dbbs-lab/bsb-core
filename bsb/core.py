from .statistics import Statistics
from .plotting import plot_network
import numpy as np
import time
from .trees import TreeCollection
from .output import MorphologyRepository
from .helpers import map_ndarray, listify_input
from .models import CellType
from .connectivity import ConnectionStrategy
from warnings import warn as std_warn
from .exceptions import *
from .reporting import report, warn, has_mpi_installed, get_report_file
from .config import JSONConfig
import json
import contextlib

###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads geometries, morphologies, ...
#    * Creates network architecture
#    * Sets up simulation


class TreeCollectionGroup:
    def add_collection(self, name, handler):
        self.__dict__[name] = TreeCollection(name, handler)


def from_hdf5(file):
    """
    Generate a :class:`.core.Scaffold` from an HDF5 file.

    :param file: Path to the HDF5 file.
    :returns: A scaffold object
    :rtype: :class:`Scaffold`
    """
    from .config import _from_hdf5

    config = _from_hdf5(file)
    # Overwrite the compile-time file specification as the file might have been moved or
    # renamed.
    config.output_formatter.file = file
    # If a morphology_repository specification is present, remove it. Standalone
    # morphology repositories are integrated  into the HDF5 file during compilation.
    if hasattr(config.output_formatter, "morphology_repository"):
        del config.output_formatter.morphology_repository
    return Scaffold(config, from_file=file)


class Scaffold:
    """
    This is the main object of the bsb package and bootstraps itself
    with a :doc:`configuration </configuration>`.

    During the compilation phase it can :doc:`place </placement>` and
    :doc:`connect </connectivity>` cells based on Layers,
    :doc:`cell type </guides/cell-type>` and :doc:`connection type
    </guides/connection-type>` configuration.

    The output can be stored in different :doc:`formats </guides/formats>` and
    can be used to have the scaffold set up simulations in common neuroscience
    simulators such as NEST or NEURON.
    """

    def __init__(self, config, from_file=None):
        self._initialise_MPI()
        self.configuration = config
        self.reset_network_cache()
        # Debug statistics, unused.
        self.statistics = Statistics(self)
        self._initialise_output_formatter()
        self.trees = TreeCollectionGroup()
        self.trees.add_collection("cells", self.output_formatter)
        self.trees.add_collection("morphologies", self.output_formatter)
        self._nextId = 0
        # Use the configuration to initialise all components such as cells and layers
        # to prepare for the network architecture compilation.
        self._intialise_components()
        self._intialise_simulators()

        # Tell the output formatter that we've loaded from an output and initialise scaffold from it.
        if from_file:
            self.output_formatter.file = from_file
            self.output_formatter.init_scaffold()

    def _initialise_MPI(self):
        # Delegate initialization of MPI to the reporting module. Which is weird, bu
        # required to make NEURON play nice. Check the results here and copy them over.
        # `has_mpi_installed` is imported from the `.reporting` namespace.
        if has_mpi_installed:
            # Import mpi4py and its MPI submodule.
            from mpi4py import MPI

            self.MPI = MPI
            self.MPI_rank = MPI.COMM_WORLD.rank
            self.has_mpi_installed = True
            self.is_mpi_master = self.MPI_rank == 0
            self.is_mpi_slave = self.MPI_rank != 0
        else:
            self.has_mpi_installed = False
            self.is_mpi_master = True
            self.is_mpi_slave = False

    def _intialise_components(self):
        """
        Initialise all of the components of the scaffold.

        The configuration step parses the configuration string into scaffold
        components such as layers, cell types, simulations, ... The initialisation
        step then does the following:

        * Hand each component a reference to the scaffold they're a part of.
        * Run the component specific validation (call `component.validate`)
        * Boot each component.
        """
        # Initialise the components now that the scaffoldInstance is available
        self._initialise_layers()
        self._initialise_cells()
        self._initialise_morphologies()
        self._initialise_placement_strategies()
        self._initialise_connection_types()
        self._initialise_simulations()
        self._initialise_hooks()

    def report(self, message, level=2, ongoing=False, token=None):
        """
        Send a message to the appropriate output channel.

        :param message: Text message to send.
        :type message: string
        :param level: Verbosity level of the message.
        :type level: int
        :param ongoing: The message is part of an ongoing progress report. This replaces the endline (`\\n`) character with a carriage return (`\\r`) character
        :deprecated: Use :func:`.reporting.report`
        """
        std_warn("Deprecated in favor of `bsb.reporting.report`.", UserDeprecationWarning)
        report(message, level=level, ongoing=ongoing)

    def warn(self, message, category=None):
        """
        Send a warning.

        :param message: Warning message
        :type message: string
        :param category: The class of the warning.
        :deprecated: Use :func:`.reporting.warn`
        """
        std_warn("Deprecated in favor of `bsb.reporting.warn`.", UserDeprecationWarning)
        warn(message, category)

    def _intialise_simulators(self):
        self.simulators = self.configuration.simulators

    def _initialise_cells(self):
        for cell_type in self.configuration.cell_types.values():
            cell_type.initialise(self)

    def _initialise_layers(self):
        for layer in self.configuration.layers.values():
            layer.initialise(self)

    def _initialise_placement_strategies(self):
        for placement in self.configuration.placement_strategies.values():
            placement.initialise(self)

    def _initialise_connection_types(self):
        for connection_type in self.configuration.connection_types.values():
            connection_type.initialise(self)
            # Wrap the connect function.
            connection_type._wrap_connect()

    def _initialise_morphologies(self):
        for geometry in self.configuration.morphologies.values():
            geometry.initialise(self)

    def _initialise_hooks(self):
        for hook in self.configuration.after_placement_hooks.values():
            hook.initialise(self)
        for hook in self.configuration.after_connect_hooks.values():
            hook.initialise(self)

    def _initialise_simulations(self):
        for simulation in self.configuration.simulations.values():
            self._initialise_simulation(simulation)

    def _initialise_simulation(self, simulation):
        simulation.initialise(self)
        for sim_cell in simulation.cell_models.values():
            sim_cell.initialise(self)
        for sim_connection in simulation.connection_models.values():
            sim_connection.initialise(self)
        for device in simulation.devices.values():
            device.initialise(self)

    def place_cell_types(self):
        """
        Run the placement strategies of all cell types.
        """
        sorted_cell_types = CellType.resolve_order(self.configuration.cell_types)
        for cell_type in sorted_cell_types:
            self.place_cell_type(cell_type)

    def place_cell_type(self, cell_type):
        """
        Place a cell type.
        """
        # Place cell type according to PlacementStrategy
        cell_type.placement.place()
        if cell_type.entity:
            entities = self.entities_by_type[cell_type.name]
            report(
                "Finished placing {} {} entities.".format(len(entities), cell_type.name),
                level=2,
            )
        else:
            # Get the placed cells
            cells = self.cells_by_type[cell_type.name][:, 2:5]
            # Construct a tree of the placed cells
            self.trees.cells.create_tree(cell_type.name, cells)
            report(
                "Finished placing {} {} cells.".format(len(cells), cell_type.name),
                level=2,
            )

    def connect_cell_types(self):
        """
        Run the connection strategies of all cell types.
        """
        sorted_connection_types = ConnectionStrategy.resolve_order(
            self.configuration.connection_types
        )
        for connection_type in sorted_connection_types:
            self.connect_type(connection_type)

    def connect_type(self, connection_type):
        """
        Run a connection type
        """
        source_name = connection_type.from_cell_types[0].name
        target_name = connection_type.to_cell_types[0].name
        report(
            "Started connecting {} with {} .".format(source_name, target_name),
            level=2,
        )
        connection_type.connect()
        # Iterates for each tag of the connection_type
        for tag in range(len(connection_type.tags)):

            conn_num = np.shape(connection_type.get_connection_matrices()[tag])[0]
            report(
                "Finished connecting {} with {} (tag: {} - total connections: {}).".format(
                    source_name, target_name, connection_type.tags[tag], conn_num
                ),
                level=2,
            )

    def run_after_placement_hooks(self):
        """
        Run all after placement hooks.
        """
        for hook in self.configuration.after_placement_hooks.values():
            hook.after_placement()

    def run_after_connectivity_hooks(self):
        """
        Run all after placement hooks.
        """
        for hook in self.configuration.after_connect_hooks.values():
            hook.after_connectivity()

    def compile_network(self, tries=1, output=True):
        """
        Run all steps in the scaffold sequence to obtain a full network.

        :param output: Store the network after compilation.
        :type output: boolean
        """
        times = np.zeros(tries)
        for i in np.arange(tries, dtype=int):
            if i > 0:
                self.reset_network_cache()
            t = time.time()
            for step in (
                self.place_cell_types,
                self.run_after_placement_hooks,
                self.connect_cell_types,
                self.run_after_connectivity_hooks,
            ):
                step()
                if output:
                    if not has_mpi_installed:
                        self.compile_output()
                    else:
                        if self.is_mpi_master:
                            self.compile_output()
                            self.MPI.COMM_WORLD.bcast(self.output_formatter.file, root=0)
                        else:
                            warn(
                                "Distributed compiling under MPI is not possible."
                                + "All nodes except the master node are waiting, "
                                + "doing nothing. Please compile on a single node.",
                                ResourceWarning,
                            )
                            self.output_formatter.file = self.MPI.COMM_WORLD.bcast(
                                None, root=0
                            )

            for type in self.configuration.cell_types.values():
                if type.entity:
                    count = self.entities_by_type[type.name].shape[0]
                else:
                    count = self.cells_by_type[type.name].shape[0]
                placed = type.placement.get_placement_count()
                if placed == 0 or count == 0:
                    report("0 {} placed (0%)".format(type.name), level=1)
                    continue
                density_msg = ""
                percent = int((count / type.placement.get_placement_count()) * 100)
                if type.placement.layer is not None:
                    volume = type.placement.layer_instance.volume
                    density_gotten = "%.4g" % (count / volume)
                    density_wanted = "%.4g" % (
                        type.placement.get_placement_count() / volume
                    )
                    density_msg = " Desired density: {}. Actual density: {}".format(
                        density_wanted, density_gotten
                    )
                report(
                    "{} {} placed ({}%).".format(
                        count,
                        type.name,
                        percent,
                    ),
                    level=2,
                )
            report("Average runtime: {}".format(np.average(times)), level=2)

    def _initialise_output_formatter(self):
        self.output_formatter = self.configuration.output_formatter
        self.output_formatter.initialise(self)
        # Alias the output formatter to some other functions it provides.
        self.morphology_repository = self.output_formatter
        self.tree_handler = self.output_formatter
        # Load an actual morphology repository if it is provided
        if (
            not self.is_compiled()
            and self.output_formatter.morphology_repository is not None
        ):
            # We are in a precompilation state and the configuration specifies us to use a morpho repo.
            self.morphology_repository = MorphologyRepository(
                self.output_formatter.morphology_repository
            )

    def plot_network_cache(self, fig=None):
        """
        Plot everything currently in the network cache.
        """
        plot_network(self, fig=fig, from_memory=True)

    def reset_network_cache(self):
        """
        Clear out everything stored in the network cache.
        """
        # Cell positions dictionary per cell type. Columns: X, Y, Z.
        cell_types = list(
            filter(
                lambda c: not hasattr(c, "entity") or not c.entity,
                self.configuration.cell_types.values(),
            )
        )
        entities = list(
            filter(
                lambda c: hasattr(c, "entity") and c.entity,
                self.configuration.cell_types.values(),
            )
        )
        self.cells_by_type = {c.name: np.empty((0, 5)) for c in cell_types}
        # Entity IDs per cell type.
        self.entities_by_type = {e.name: np.empty((0)) for e in entities}
        # Cell connections per connection type. Columns: From ID, To ID.
        self.cell_connections_by_tag = {
            key: np.empty((0, 2)) for key in self.configuration.connection_types.keys()
        }
        self.connection_morphologies = {}
        self.connection_compartments = {}
        self.appends = {}
        self._connectivity_set_meta = {}
        self.labels = {}
        self.rotations = {}

    def run_simulation(self, simulation_name, quit=False):
        """
        Run a simulation starting from the default single-instance adapter.

        :param simulation_name: Name of the simulation in the configuration.
        :type simulation_name: string
        """
        t = time.time()
        simulation, simulator = self.prepare_simulation(simulation_name)
        # If we're reporting to a file, add a stream of progress event messages..
        report_file = get_report_file()
        if report_file:
            listener = ReportListener(self, report_file)
            simulation.add_progress_listener(listener)
        simulation.simulate(simulator)
        result_path = simulation.collect_output(simulator)
        time_sim = time.time() - t
        report("Simulation runtime: {}".format(time_sim), level=2)
        if quit and hasattr(simulator, "quit"):
            simulator.quit()
        return result_path

    def get_simulation(self, simulation_name):
        """
        Retrieve the default single-instance adapter for a simulation.
        """
        if simulation_name not in self.configuration.simulations:
            raise SimulationNotFoundError(
                "Unknown simulation '{}', choose from: {}".format(
                    simulation_name, ", ".join(self.configuration.simulations.keys())
                )
            )
        simulation = self.configuration.simulations[simulation_name]
        return simulation

    def prepare_simulation(self, simulation_name):
        """
        Retrieve and prepare the default single-instance adapter for a simulation.
        """
        simulation = self.get_simulation(simulation_name)
        simulator = simulation.prepare()
        return simulation, simulator

    def place_cells(self, cell_type, layer, positions, rotations=None):
        """
        Place cells inside of the scaffold

        .. code-block:: python

            # Add one granule cell at position 0, 0, 0
            cell_type = scaffold.get_cell_type("granule_cell")
            scaffold.place_cells(cell_type, cell_type.layer_instance, [[0., 0., 0.]])

        :param cell_type: The type of the cells to place.
        :type cell_type: :class:`.models.CellType`
        :param layer: The layer in which to place the cells.
        :type layer: :class:`.models.Layer`
        :param positions: A collection of xyz positions to place the cells on.
        :type positions: Any `np.concatenate` type of shape (N, 3).
        """
        cell_count = positions.shape[0]
        if cell_count == 0:
            return
        # Create an ID for each cell.
        cell_ids = self._allocate_ids(positions.shape[0])
        # Spoof old cache
        cell_data = np.column_stack((cell_ids, np.zeros(positions.shape[0]), positions))
        # Cache them per type
        self.cells_by_type[cell_type.name] = np.concatenate(
            (self.cells_by_type[cell_type.name], cell_data)
        )

        placement_dict = self.statistics.cells_placed
        if cell_type.name not in placement_dict:
            placement_dict[cell_type.name] = 0
        placement_dict[cell_type.name] += cell_count
        if not hasattr(cell_type.placement, "cells_placed"):
            setattr(cell_type.placement, "cells_placed", 0)
        cell_type.placement.cells_placed += cell_count

        if rotations is not None:
            if cell_type.name not in self.rotations:
                self.rotations[cell_type.name] = np.empty((0, 2))
            self.rotations[cell_type.name] = np.concatenate(
                (self.rotations[cell_type.name], rotations)
            )
        return cell_ids

        return cell_ids

    def _allocate_ids(self, count):
        # Allocate a set of unique cell IDs in the scaffold.
        IDs = np.array(range(self._nextId, self._nextId + count), dtype=int)
        self._nextId += count
        return IDs

    def connect_cells(
        self,
        connection_type,
        connectome_data,
        tag=None,
        morphologies=None,
        compartments=None,
        meta=None,
        morpho_map=None,
    ):
        """
        Store connections for a connection type. Will store the
        ``connectome_data`` under ``bsb.cell_connections_by_tag``, a
        mapped version of the morphology names under
        ``bsb.connection_morphologies`` and the compartments under
        ``bsb.connection_compartments``.

        :param connection_type: The connection type. The name of the connection type will be used by default as the tag.
        :type connection_type: :class:`ConnectionStrategy`
        :param connectome_data: A 2D ndarray with 2 columns: the presynaptic cell id and the postsynaptic cell id.
        :type connectome_data: :class:`numpy.ndarray`
        :param tag: The name of the dataset in the storage. If no tag is given, the name of the connection type is used. This parameter can be used to create multiple different connection set per connection type.
        :type tag: string
        :param morphologies: A 2D ndarray with 2 columns: the presynaptic morphology name and the postsynaptic morphology name.
        :type morphologies: :class:`numpy.ndarray`
        :param compartments: A 2D ndarray with 2 columns: the presynaptic compartment id and the postsynaptic compartment id.
        :type compartments: :class:`numpy.ndarray`
        :param meta: Additional metadata to be stored on the connectivity set.
        :type meta: dict
        """
        # Some array preprocessing
        if not isinstance(connectome_data, np.ndarray):
            connectome_data = np.array(connectome_data)
        if len(connectome_data.shape) == 1:
            connectome_data = connectome_data.reshape(-1, 2)
        # Allow 1 connection type to store multiple connectivity datasets by utilizing tags
        tag = tag or connection_type.name
        # Keep track of relevant tags in the connection_type object
        if tag not in connection_type.tags:
            connection_type.tags.append(tag)
        self._append_tagged("cell_connections_by_tag", tag, connectome_data)
        if compartments is not None or morphologies is not None:
            if len(morphologies) != len(connectome_data) or len(compartments) != len(
                connectome_data
            ):
                raise MorphologyDataError(
                    "The morphological data did not match the connectome data."
                )
            self._append_mapped(
                "connection_morphologies", tag, morphologies, use_map=morpho_map
            )
            self._append_tagged("connection_compartments", tag, compartments)
        # Store the metadata internally until the output is compiled.
        if meta is not None:
            self._connectivity_set_meta[tag] = meta

    def create_entities(self, cell_type, count):
        """
        Create entities in the simulation space.

        Entities are different from cells because they have no positional data and
        don't influence the placement step. They do have a representation in the
        connection and simulation step.

        :param cell_type: The cell type of the entities
        :type cell_type: :class:`.models.CellType`
        :param count: Number of entities to place
        :type count: int
        """
        if count == 0:
            return
        # Create an ID for each entity.
        entities_ids = self._allocate_ids(count)

        # Cache them per type
        if not cell_type.name in self.entities_by_type:
            self.entities_by_type[cell_type.name] = entities_ids
        else:
            self.entities_by_type[cell_type.name] = np.concatenate(
                (self.entities_by_type[cell_type.name], entities_ids)
            )

        placement_dict = self.statistics.cells_placed
        if not cell_type.name in placement_dict:
            placement_dict[cell_type.name] = 0
        placement_dict[cell_type.name] += count
        if not hasattr(cell_type.placement, "cells_placed"):
            setattr(cell_type.placement, "cells_placed", 0)
        cell_type.placement.cells_placed += count

        return entities_ids

    def _append_tagged(self, attr, tag, data):
        """
        Appends or creates data to a tagged numpy array in a dictionary attribute of
        the scaffold.
        """
        if tag in self.__dict__[attr]:
            cache = self.__dict__[attr][tag]
            self.__dict__[attr][tag] = np.concatenate((cache, data))
        else:
            self.__dict__[attr][tag] = np.copy(data)

    def _append_mapped(self, attr, tag, data, use_map=None):
        """
        Appends or creates the data with a map to a tagged numpy array in a dictionary
        attribute of the scaffold.
        """
        attr_data = getattr(self, attr)
        map_name = f"__map_{tag}"
        if map_name not in attr_data:
            attr_data[map_name] = []
        # Map data
        if use_map:
            data_map = use_map
            if len(data):
                # Using `+` on empty dataset errors
                data += len(attr_data[map_name])
            mapped_data = np.array(data, dtype=int)
        else:
            if data.dtype.type is np.string_:
                # Explicitly cast numpy strings to str so they don't yield
                # `b'morphology_name'` when stored as attribute by hdf5.
                data = data.astype(str)
            mapped_data, data_map = map_ndarray(data, _map=attr_data[map_name])
            mapped_data = np.array(mapped_data, dtype=int)
        attr_data[map_name].extend(data_map)

        # Append data
        if tag in attr_data:
            cache = attr_data[tag]
            attr_data[tag] = np.concatenate((cache, mapped_data))
        else:
            attr_data[tag] = np.copy(mapped_data)

    def append_dset(self, name, data):
        """
        Append a custom dataset to the scaffold output.

        :param name: Unique identifier for the dataset.
        :type name: string
        :param data: The dataset
        """
        self.appends[name] = data

    def load_appendix(self, name, skip_cache=False):
        """
        Load a custom dataset from the scaffold cache or output.

        :param name: Unique identifier for the dataset.
        :type name: string
        :param skip_cache: Ignore any cached data and read only from the output.
        :type skip_cache: bool
        """
        if not skip_cache and name in self.appends:
            return self.appends[name]
        else:
            return self.output_formatter.load_appendix(name)

    def get_cells_by_type(self, name):
        """
        Find all of the cells of a certain type. This information will be gathered
        from the cache first, and if that isn't present, from persistent storage.

        :param name: Name of the cell type.
        """
        if name not in self.cells_by_type:
            raise TypeNotFoundError(
                "Attempting to load unknown cell type '{}'".format(name)
            )
        if self.cells_by_type[name].shape[0] == 0:
            if not self.output_formatter.exists():
                return self.cells_by_type[name]
            if self.output_formatter.has_cells_of_type(name):
                self.cells_by_type[name] = self.output_formatter.get_cells_of_type(name)
            else:
                raise TypeNotFoundError(
                    "Cell type '{}' not found in output storage".format(name)
                )
        return self.cells_by_type[name]

    def get_entities_by_type(self, name):
        """
        Find all of the entities of a certain type. This information will be gathered
        from the cache first, and if that isn't present, from persistent storage.

        :param name: Name of the cell type.
        """
        if name not in self.entities_by_type:
            raise TypeNotFoundError(
                "Attempting to load unknown entity type '{}'".format(name)
            )
        if self.entities_by_type[name].shape[0] == 0:
            if not self.output_formatter.exists():
                return self.entities_by_type[name]
            if self.output_formatter.has_cells_of_type(name, entity=True):
                self.entities_by_type[name] = self.output_formatter.get_cells_of_type(
                    name, entity=True
                )
            else:
                raise TypeNotFoundError(
                    "Entity type '{}' not found in output storage".format(name)
                )
        return self.entities_by_type[name]

    def compile_output(self):
        """
        Task the output formatter to generate all output from the current state.

        The default formatter is the HDF5Formatter; therefor calling this function
        will generate an HDF5 file that contains all the data currently present in
        this object.
        """
        self.output_formatter.create_output()

    def partial_placement(self, place_types, append=False):
        if append:
            raise NotImplementedError(
                "Coming in v4. Open an issue on GitHub if you require partial (re)placement with append before v4"
            )
        raise NotImplementedError(
            "Coming in v4. Open an issue on GitHub if you require partial (re)placement before v4"
        )

    @contextlib.contextmanager
    def partial_connect(self, conn_tags, append=False):
        """
        Creates a context in which you can execute connection strategies and will
        partially compile only the given ``conn_tags`` afterwards.

        Example
        -------

        .. code-block:: python

          with network.partial_connect(["a_to_b", "b_to_c"]):
            network.connection_types["a_to_b"].connect()
            network.connection_types["b_to_c"].connect()

        :param conn_tags: The connection **tags** to write to output. Each
          connection type that you execute may produce 0, 1 or more tags.
        :type conn_tags: List[str]
        """
        if append:
            raise NotImplementedError(
                "Coming in v4. Open an issue on GitHub if you require partial"
                + "(re)connects with append before v4."
            )
        oc = self.cell_connections_by_tag
        self.cell_connections_by_tag = {
            cnt: np.zeros((0, 2), dtype=float)
            for cnt, data in oc.items()
            if cnt in conn_tags
        }
        warn(
            "Temporary workaround (fix in v4) for partial connect of:"
            + ", ".join(self.cell_connections_by_tag.keys()),
        )
        warn(
            "Read data with `PlacementSet` and `ConnectivitySet`,"
            + " do not use `cells_by_type` or `cell_connections_by_tag`!"
        )
        warn("Write data with `connect_cells`.")
        yield
        with self.output_formatter.load("a") as f:
            f = f()
            for tag in self.cell_connections_by_tag.keys():
                for delgroup in (
                    f"/cells/connections/{tag}",
                    f"/cells/connection_compartments/{tag}",
                    f"/cells/connection_morphologies/{tag}",
                ):
                    with contextlib.suppress(KeyError):
                        del f[delgroup]
            self.output_formatter.store_cell_connections(f["/cells"])

    def _connection_types_query(self, pre_query=[], post_query=[]):
        # Filter network connection types for any type that satisfies both
        # the presynaptic and postsynaptic query. Empty queries satisfy all
        # types. The presynaptic query is satisfied if the conn type contains
        # any of the queried cell types presynaptically, and same for post.

        def partial_query(types, query):
            return not query or any(cell_type in query for cell_type in types)

        def query(conn_type):
            pre_match = partial_query(conn_type.from_cell_types, pre_query)
            post_match = partial_query(conn_type.to_cell_types, post_query)
            return pre_match and post_match

        types = self.configuration.connection_types.values()
        return [*filter(query, types)]

    def query_connection_types(self, any=None, pre=None, post=None):
        """
        Search for connection types that include specific cell types as pre- or
        postsynaptic targets.

        :param any: Cell type names that will include connection types that
          have the given cell types as either pre- or postsynaptic targets.
        :type any: string or sequence of strings.
        :param postsynaptic: Cell type names that will include connection types
          that have the given cell types as postsynaptic targets.
        :type postsynaptic: Union[CellType, List[CellType]].
        :param presynaptic: Cell type names that will include connection types
          that have the given cell types as presynaptic targets.
        :type presynaptic: Union[CellType, List[CellType]].
        :returns: The connection types that meet the specified criteria.
        :rtype: dict
        """
        if any is None and pre is None and post is None:
            raise ArgumentError("No query specified")
        pre = listify_input(pre)
        post = listify_input(post)
        if any is not None:
            any = listify_input(any)
            pre.extend(any)
            post.extend(any)

        return self._connection_types_query(pre, post)

    def query_connection_cache(self, any=None, pre=None, post=None):
        """
        Get the connections currently in the cache for connection types that
        include certain cell types as targets.

        :param any: Cell type names that will include connection types that have
          the given cell types as either pre- or postsynaptic targets.
        :type any: string or sequence of strings.
        :param postsynaptic: Cell type names that will include connection types
          that have the given cell types as postsynaptic targets.
        :type postsynaptic: Union[CellType, List[CellType]].
        :param presynaptic: Cell type names that will include connection types
          that have the given cell types as presynaptic targets.
        :type presynaptic: Union[CellType, List[CellType]].

        :see: query_connection_types
        """
        queried = self.query_connection_types(any, pre, post)
        return {type: type.get_connection_matrices() for type in queried}

    def query_connection_sets(self, any=None, pre=None, post=None):
        """
        Get the connectivity sets from storage for connection types that include
        certain cell types as targets.

        :param any: Cell type names that will include connection types that have
          the given cell types as either pre- or postsynaptic targets.
        :type any: string or sequence of strings.
        :param postsynaptic: Cell type names that will include connection types
          that have the given cell types as postsynaptic targets.
        :type postsynaptic: Union[CellType, List[CellType]].
        :param presynaptic: Cell type names that will include connection types
          that have the given cell types as presynaptic targets.
        :type presynaptic: Union[CellType, List[CellType]].

        :see: query_connection_types
        :rtype: :class:`bsb.models.ConnectivitySet`
        """
        queried = self.query_connection_types(any, pre, post)
        return {type: type.get_connection_sets() for type in queried}

    def get_connectivity_sets(self):
        """
        Return all connectivity sets from the output formatter.

        :param tag: Unique identifier of the connectivity set in the output formatter
        :type tag: string
        :returns: A connectivity set
        :rtype: :class:`.models.ConnectivitySet`
        """
        return self.output_formatter.get_connectivity_sets()

    def get_connectivity_set(self, tag):
        """
        Return a connectivity set from the output formatter.

        :param tag: Unique identifier of the connectivity set in the output formatter
        :type tag: string
        :returns: A connectivity set
        :rtype: :class:`.models.ConnectivitySet`
        """
        return self.output_formatter.get_connectivity_set(tag)

    def get_placement_set(self, type, labels=None):
        """
        Return a cell type's placement set from the output formatter.

        :param type: Unique identifier of the cell type in the scaffold.
        :type type: :class:`.models.CellType` or string
        :returns: A placement set
        :rtype: :class:`.models.PlacementSet`
        """
        if isinstance(type, str):
            type = self.get_cell_type(type)
        ps = self.output_formatter.get_placement_set(type)
        if labels is not None:

            def label_filter():
                return np.concatenate(tuple(self.get_labelled_ids(l) for l in labels))

            ps.set_filter(label_filter)
        return ps

    def translate_cell_ids(self, data, cell_type):
        """
        Return the global ids of the N-th cells of a cell type

        .. code-block:: python

            cell_type = scaffold.get_cell_type('granule_cell')
            # Get the global ids of the first 3 granule cells.
            global_ids = scaffold.translate_cell_ids([0, 1, 2], cell_type)

        .. code-block::

            >>> [1312, 1313, 1314]

        :param data: A valid index for a :class:`numpy.ndarray`
        :param cell_type: A cell type.
        :type cell_type: :class:`.models.CellType`
        """
        if not self.is_compiled():
            return self.cells_by_type[cell_type.name][data, 0]
        else:
            return self.get_placement_set(cell_type).identifiers[data]

    def get_connection_type(self, name):
        """
        Get the specified connection type.

        :param name: Unique identifier of the connection type in the configuration.
        :type name: string

        :returns: The connection type
        :rtype: :class:`.connectivity.ConnectionStrategy`
        :raise TypeNotFoundError: When the specified name is not known.
        """
        if name not in self.configuration.connection_types:
            raise TypeNotFoundError("Unknown connection type '{}'".format(name))
        return self.configuration.connection_types[name]

    def get_cell_types(self, entities=True):
        """
        Return a collection of all configured cell types.

        ::

          for cell_type in scaffold.get_cell_types():
              print(cell_type.name)
        """
        if entities:
            return list(self.configuration.cell_types.values())
        else:
            return [c for c in self.get_cell_types() if not c.entity]

    def get_entity_types(self):
        """
        Return a list of connection types that describe entities instead
        of cells.
        """
        return list(
            filter(
                lambda t: hasattr(t, "entity") and t.entity is True,
                self.configuration.connection_types.values(),
            )
        )

    def get_cell_type(self, identifier):
        """
        Return the specified cell type.

        :param identifier: Unique identifier of the cell type in the configuration,
          either its name or ID.
        :type identifier: string (name) or int (ID)
        :returns: The cell type
        :rtype: :class:`.models.CellType`
        :raise TypeNotFoundError: When the specified identifier is not known.
        """
        return self.configuration.get_cell_type(identifier)

    def assert_continuity(self, gaps_ok=False):
        """
        Assert that all PlacementSets consist of only 1 continuous stretch of IDs, and
          that all PlacementSets follow
        each other without gaps, starting from zero.

        :param gaps_ok: Check that just the cell types are continuous, but allow gaps
          between them.
        :type gaps_ok: bool
        """
        beginnings = set()
        ends = dict()
        for ct in self.get_cell_types():
            stretch = ct.get_placement_set()._identifiers.get_dataset()
            if len(stretch) != 2:
                raise ContinuityError(
                    f"Discontinuities in `{ct.name}`:"
                    + " multiple ID stretches in a single placement set."
                )
            beginnings.add(stretch[0])
            # Adding the count to the beginning gives the ID with which another
            # set should begin.
            ends[ct.name] = stretch[0] + stretch[1]
        if gaps_ok:
            return True
        if 0 not in beginnings:
            raise ContinuityError("Placement data does not start at ID 0.")
        loose_ends = []
        # Since the ends should be the beginning of exactly 1 other set we remove each end
        # from the beginnings list. If this happens twice we get a KeyError, or if the
        # beginning never existed. Mark those as a loose end, if there is not exactly 1
        # loose end, there is some branching, gaps or overlap.
        for name, end in ends.items():
            try:
                beginnings.remove(end)
            except KeyError:
                loose_ends.append(name)
        if len(loose_ends) != 1:
            raise ContinuityError(
                "Discontinuous ends detected: " + ", ".join(loose_ends) + "."
            )
        return True

    def get_gid_types(self, ids):
        """
        Return the cell type of each gid
        """
        all_ps = {
            ct: self.get_placement_set(ct).identifiers
            for ct in self.configuration.cell_types.values()
        }

        def lookup(id):
            for ct, ps in all_ps.items():
                if id in ps:
                    return ct

        return np.vectorize(lookup)(ids)

    def get_placed_count(self, cell_type_name):
        """
        Return the amount of cell of a cell type placed in the volume.

        :param cell_type_name: Unique identifier of the cell type in the configuration.
        :type cell_type_name: string
        """
        return self.statistics.cells_placed[cell_type_name]

    def is_compiled(self):
        """
        Returns whether there persistent storage of this network has been created.

        :rtype: boolean
        """
        return self.output_formatter.exists()

    def create_adapter(self, simulation_name):
        """
        Create an adapter for a simulation. Adapters are the objects that translate
        scaffold data into simulator data.
        """
        if simulation_name not in self.configuration.simulations:
            raise SimulationNotFoundError(
                "Unknown simulation '{}'".format(simulation_name)
            )
        simulations = self.configuration._parsed_config["simulations"]
        simulation_config = simulations[simulation_name]
        adapter = self.configuration.init_simulation(
            simulation_name, simulation_config, return_obj=True
        )
        self.configuration.finalize_simulation(
            simulation_name, simulation_config, adapter
        )
        self._initialise_simulation(adapter)
        return adapter

    def label_cells(self, ids, label):
        """
        Store labels for the given cells. Labels can be used to identify subsets of cells.

        :param ids: global identifiers of the cells that need to be labelled.
        :type ids: iterable
        """
        # Initialize (if required)
        if not label in self.labels.keys():
            self.labels[label] = []
        # Extend labels
        self.labels[label].extend(ids)

    def get_labels(self, pattern):
        """
        Retrieve the set of labels that match a label pattern. Currently only exact
        matches or strings ending in a wildcard are supported:

        .. code-block:: python

            # Will return only ["label-53"] if it is known to the scaffold.
            labels = scaffold.get_labels("label-53")
            # Might return multiple labels such as ["label-53", "label-01", ...]
            labels = scaffold.get_labels("label-*")

        :param pattern: An exact match or pattern ending in a wildcard (*) character.
        :type pattern: string

        :returns: All labels matching the pattern
        :rtype: list
        """
        if pattern.endswith("*"):
            p = pattern[:-1]
            finder = lambda l: l.startswith(p)
        else:
            finder = lambda l: l == pattern
        return list(filter(finder, self.labels.keys()))

    def get_labelled_ids(self, label):
        """
        Get all the global identifiers of cells labelled with the specific label.
        """
        try:
            data = self.labels[label]
        except KeyError:
            data = []
        return np.array(data, dtype=int)

    def get_cell_total(self):
        """
        Return the total amount of cells and entities placed.
        """
        return sum(list(self.statistics.cells_placed.values()))

    def for_blender(self):
        """
        Binds all blender functions onto the scaffold object.
        """
        from .blender import _mixin

        for f_name, f in _mixin.__dict__.items():
            if callable(f) and not f_name.startswith("_"):
                self.__dict__[f_name] = f.__get__(self)

        return self

    def left_join(self, other, label=None):
        """
        Joins cell placement and cell connectivity of a new scaffold object
        into self scaffold object.

        If label is not None the cells of coming from the original
        and the new scaffold will be labelled differently in the merged scaffold.

        """

        id_map = {}
        for ct in self.get_cell_types():
            for c in other.get_cell_types():
                if c.name != ct.name:
                    continue
                ps = other.get_placement_set(c)
                old_ids = ps.identifiers
                if not ct.entity:
                    ids = self.place_cells(ct, ct.placement.layer_instance, ps.positions)
                else:
                    ids = self.create_entities(ct, len(ids))
                id_map[c.name] = dict(zip(old_ids, ids))
                if label is not None:
                    self.label_cells(ids, label)
        for ct_self in self.configuration.connection_types.values():
            missing = True
            for ct_other in other.configuration.connection_types.values():
                from_type = ct_other.from_cell_types[0]
                to_type = ct_other.to_cell_types[0]
                if ct_self.name != ct_other.name:
                    continue
                missing = False
                conn_set = other.get_connectivity_set(ct_other.name)
                if not len(conn_set):
                    break
                from_ids = conn_set.from_identifiers
                mapped_from_ids = np.vectorize(id_map[from_type.name].get)(from_ids)
                to_ids = conn_set.to_identifiers
                mapped_to_ids = np.vectorize(id_map[to_type.name].get)(to_ids)
                cds = np.column_stack((from_ids, to_ids))
                mapped_cds = np.column_stack((mapped_from_ids, mapped_to_ids))
                try:
                    comp_data = conn_set.compartment_set.get_dataset()
                    morpho_data = conn_set.morphology_set.get_dataset()
                except DatasetNotFoundError:
                    comp_data = None
                    morpho_data = None
                self.connect_cells(
                    ct_self, mapped_cds, morphologies=morpho_data, compartments=comp_data
                )
            if missing:
                raise RuntimeError(f"Missing '{ct_self}' dataset.")

        self.compile_output()
        return self


def merge(output_file, *others, label_prefix="merged_"):
    """
    Merges several scaffolds into one joining them one at time.

    :param output_file: name under which the merged scaffold will be saved
    :type output_file: string
    :param others: scaffolds that have to be merged together
    :type others: list
    """

    cfg_json = json.loads(others[0].configuration._raw)
    cfg_json["output"]["file"] = output_file
    cfg_copy = JSONConfig(stream=json.dumps(cfg_json))
    merged = Scaffold(cfg_copy)
    merged.output_formatter.create_output()

    for counter, other in enumerate(others):
        merged.left_join(other, label=f"{label_prefix}{counter}")
    return merged


def get_mrepo(file):
    """
    Shortcut function to create :class:`.output.MorphologyRepository`
    """
    return MorphologyRepository(file)


class ReportListener:
    def __init__(self, scaffold, file):
        self.file = file
        self.scaffold = scaffold

    def __call__(self, progress):
        report(
            str(progress.progression)
            + "+"
            + str(progress.duration)
            + "+"
            + str(progress.time),
            token="simulation_progress",
        )


def register_cell_targetting(name, f):
    from .simulation.targetting import TargetsNeurons

    setattr(TargetsNeurons, f"_targets_{name}", f)


def register_section_targetting(name, f):
    from .simulation.targetting import TargetsSections

    setattr(TargetsSections, f"_section_target_{name}", f)
