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
        connection_type.connect()
        # Iterates for each tag of the connection_type
        for tag in range(len(connection_type.tags)):
            conn_num = np.shape(connection_type.get_connection_matrices()[tag])[0]
            source_name = connection_type.from_cell_types[0].name
            target_name = connection_type.to_cell_types[0].name
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
            self.place_cell_types()
            self.run_after_placement_hooks()
            if output:
                self.compile_output()
            self.connect_cell_types()
            self.run_after_connectivity_hooks()
            times[i] = time.time() - t

            if output:
                self.compile_output()

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
        # Cell positions dictionary per layer. Columns: Type, X, Y, Z.
        self.cells_by_layer = {
            key: np.empty((0, 5)) for key in self.configuration.layers.keys()
        }
        # Cells collection. Columns: Cell ID, Type, X, Y, Z.
        self.cells = np.empty((0, 5))
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
        simulation.collect_output()
        time_sim = time.time() - t
        report("Simulation runtime: {}".format(time_sim), level=2)
        if quit and hasattr(simulator, "quit"):
            simulator.quit()
        time_sim = time.time() - t
        report("Simulation runtime: {}".format(time_sim), level=2)
        return simulation

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
            scaffold.place_cells(cell_type, cell_type.layer_istance, [[0., 0., 0.]])

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
        # Store cells as ID, typeID, X, Y, Z
        cell_data = np.column_stack(
            (cell_ids, np.ones(positions.shape[0]) * cell_type.id, positions)
        )
        # Cache them per type
        self.cells_by_type[cell_type.name] = np.concatenate(
            (self.cells_by_type[cell_type.name], cell_data)
        )
        # Cache them per layer
        self.cells_by_layer[layer.name] = np.concatenate(
            (self.cells_by_layer[layer.name], cell_data)
        )
        # Store
        self.cells = np.concatenate((self.cells, cell_data))

        placement_dict = self.statistics.cells_placed
        if cell_type.name not in placement_dict:
            placement_dict[cell_type.name] = 0
        placement_dict[cell_type.name] += cell_count
        if not hasattr(cell_type.placement, "cells_placed"):
            cell_type.placement.__dict__["cells_placed"] = 0
        cell_type.placement.cells_placed += cell_count

        if rotations is not None:
            if cell_type.name not in self.rotations:
                self.rotations[cell_type.name] = np.empty((0, 2))
            self.rotations[cell_type.name] = np.concatenate(
                (self.rotations[cell_type.name], rotations)
            )

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
            cell_type.placement.__dict__["cells_placed"] = 0
        cell_type.placement.cells_placed += count

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
        # Map data
        if use_map:  # Is the data already mapped and should we use the given map?
            if not attr + "_map" in self.__dict__[attr]:
                self.__dict__[attr][tag + "_map"] = use_map.copy()
            else:
                data += len(self.__dict__[attr][tag + "_map"])
                self.__dict__[attr][tag + "_map"].extend(use_map)
            mapped_data = np.array(data, dtype=int)
        else:
            if not attr + "_map" in self.__dict__[attr]:
                self.__dict__[attr][tag + "_map"] = []
            mapped_data, data_map = map_ndarray(
                data, _map=self.__dict__[attr][tag + "_map"]
            )
            mapped_data = np.array(mapped_data, dtype=int)

        # Append data
        if tag in self.__dict__[attr]:
            cache = self.__dict__[attr][tag]
            self.__dict__[attr][tag] = np.concatenate((cache, mapped_data))
        else:
            self.__dict__[attr][tag] = np.copy(mapped_data)

    def append_dset(self, name, data):
        """
        Append a custom dataset to the scaffold output.

        :param name: Unique identifier for the dataset.
        :type name: string
        :param data: The dataset
        """
        self.appends[name] = data

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

    def _connection_types_query(self, postsynaptic=[], presynaptic=[]):
        # This function searches through all connection types that include the given
        # pre- and/or postsynaptic cell types.

        # Make sure the inputs are lists.
        postsynaptic = listify_input(postsynaptic)
        presynaptic = listify_input(presynaptic)

        # Local function that checks for any intersection between 2 lists based on a given f.
        def any_intersect(l1, l2, f=lambda x: x):
            if not l2:  # Return True if there's no pre/post targets specified
                return True
            for e1 in l1:
                if f(e1) in l2:
                    return True
            return False

        # Extract the connection types as tuples so that they can be turned back into a
        # dictionary after filtering
        connection_items = self.configuration.connection_types.items()
        # Lambda that includes any connection type with at least one of the specified
        # presynaptic and one of the specified postsynaptic connections.
        # If the post- or presynaptic constraints are empty all connection types pass for
        # that constraint.
        intersect = lambda c: any_intersect(
            c[1].to_cell_types, postsynaptic, lambda x: x.name
        ) and any_intersect(c[1].from_cell_types, presynaptic, lambda x: x.name)
        # Filter all connection types based on the lambda function.
        filtered_connection_items = list(
            filter(
                intersect,
                connection_items,
            )
        )
        # Turn the filtered result into a dictionary.
        return dict(filtered_connection_items)

    def get_connection_types_by_cell_type(
        self, any=None, postsynaptic=None, presynaptic=None
    ):
        """
        Search for connection types that include specific cell types as pre- or postsynaptic targets.

        :param any: Cell type names that will include connection types that have the given cell types as either pre- or postsynaptic targets.
        :type any: string or sequence of strings.
        :param postsynaptic: Cell type names that will include connection types that have the given cell types as postsynaptic targets.
        :type postsynaptic: string or sequence of strings.
        :param presynaptic: Cell type names that will include connection types that have the given cell types as presynaptic targets.
        :type presynaptic: string or sequence of strings.
        :returns: The connection types that meet the specified criteria.
        :rtype: dict
        """
        if any is None and postsynaptic is None and presynaptic is None:
            raise ArgumentError("No cell types specified")
        # Make a list out of the input elements
        postsynaptic = listify_input(postsynaptic)
        presynaptic = listify_input(presynaptic)
        # Initialize empty omitted lists
        if any is not None:  # Add any cell types as both post and presynaptic targets
            any = listify_input(any)
            postsynaptic.extend(any)
            presynaptic.extend(any)
        # Execute the query and return results.
        return self._connection_types_query(postsynaptic, presynaptic)

    def get_connection_cache_by_cell_type(
        self, any=None, postsynaptic=None, presynaptic=None
    ):
        """
        Get the connections currently in the cache for connection types that include certain cell types as targets.

        :see: get_connection_types_by_cell_type
        """
        # Find the connection types that have the specified targets
        connection_types = self.get_connection_types_by_cell_type(
            any, postsynaptic, presynaptic
        )
        # Map them to a list of tuples with the 1st element the connection type
        # and the connection matrices appended behind it.
        return list(
            map(lambda x: (x, *x.get_connection_matrices()), connection_types.values())
        )

    def get_connections_by_cell_type(self, any=None, postsynaptic=None, presynaptic=None):
        """
        Get the connectivity sets from storage for connection types that include certain cell types as targets.

        :see: get_connection_types_by_cell_type
        :rtype: :class:`bsb.models.ConnectivitySet`
        """
        # Find the connection types that have the specified targets
        connection_types = self.get_connection_types_by_cell_type(
            any, postsynaptic, presynaptic
        )
        # Map them to a list of tuples with the 1st element the connection type
        # and the connection matrices appended behind it.
        return list(
            map(lambda x: (x, *x.get_connectivity_sets()), connection_types.values())
        )

    def get_connectivity_set(self, tag):
        """
        Return a connectivity set from the output formatter.

        :param tag: Unique identifier of the connectivity set in the output formatter
        :type tag: string
        :returns: A connectivity set
        :rtype: :class:`.models.ConnectivitySet`
        """
        return self.output_formatter.get_connectivity_set(tag)

    def get_placement_set(self, type):
        """
        Return a cell type's placement set from the output formatter.

        :param type: Unique identifier of the cell type in the scaffold.
        :type type: :class:`.models.CellType` or string
        :returns: A placement set
        :rtype: :class:`.models.PlacementSet`
        """
        if isinstance(type, str):
            type = self.get_cell_type(type)
        return self.output_formatter.get_placement_set(type)

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
            return np.array(self.output_formatter.get_type_map(cell_type))[data]

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
            return self.configuration.cell_types.values()
        else:
            return list(filter(lambda c: not c.entity, self.get_cell_types()))

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

        :param identifier: Unique identifier of the cell type in the configuration, either its name or ID.
        :type identifier: string (name) or int (ID)
        :returns: The cell type
        :rtype: :class:`.models.CellType`
        :raise TypeNotFoundError: When the specified identifier is not known.
        """
        return self.configuration.get_cell_type(identifier)

    def get_cell_position(self, id):
        """
        Return the position of the cells in the network cache.

        :param id: Index of the cell in the network cache. Should coincide with the global id of the cell, but this isn't guaranteed if you modify the network cache manually.
        :type id: int
        :returns: Position of the cell
        :rtype: (1, 3) shaped :class:`numpy.ndarray`
        """
        if not id < len(self.cells):
            raise DataNotFoundError(
                "Cell {} does not exist. (highest id is {})".format(
                    id, len(self.cells) - 1
                )
            )
        return self.cells[id, 2:5]

    def get_cell_positions(self, selector):
        """
        Return the positional data of the selected cells in the network cache.

        :param selector: Selects the cells from the network cache.
        :type selector: A valid :class:`numpy.ndarray` index
        :returns: Positions of the cells
        :rtype: (n, 3) shaped :class:`numpy.ndarray`
        """
        return self.cells[selector, 2:5]

    def get_cells(self, selector):
        """
        Return all data of the selected cells in the network cache.

        :param selector: Selects the cells from the network cache.
        :type selector: A valid :class:`numpy.ndarray` index
        :returns: Global id, type id and Position of the cells
        :rtype: (n, 5) shaped :class:`numpy.ndarray`
        """
        return self.cells[selector]

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
        return np.array(self.labels[label], dtype=int)

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
