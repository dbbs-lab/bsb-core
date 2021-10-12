from .statistics import Statistics
from .plotting import plot_network
import numpy as np
import time
from .trees import TreeCollection
from .output import MorphologyRepository
from .helpers import map_ndarray, listify_input
from .placement import PlacementStrategy
from .connectivity import ConnectionStrategy
from warnings import warn as std_warn
from .exceptions import *
from .reporting import report, warn, has_mpi_installed, get_report_file
from .config._config import Configuration
from ._pool import create_job_pool

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
    from .storage import Storage

    storage = Storage("hdf5", file)
    return storage.load()


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

    def __init__(self, config=None, storage=None, clear=False):
        """
        Bootstraps a network object.

        :param config: The configuration to use for this network. If it is omitted the
          :doc:`default configuration <config/default>` is used.
        :type config: :class:`.config.Configuration`
        :param storage: The storage to use to read and write data for this network. If it
          is omitted the configuration's ``Storage`` node is used to construct one.
        :type storage: :class:`.storage.Storage`
        :param clear: Start with a new network, clearing any previously stored information
        :type clear: bool
        :returns: A network object
        :rtype: :class:`.core.Scaffold`
        """
        self._initialise_MPI()
        self._bootstrap(config, storage)

        if clear:
            self.clear()

        # # Debug statistics, unused.
        # self.statistics = Statistics(self)
        # self.trees = TreeCollectionGroup()
        # self.trees.add_collection("cells", self.storage)
        # self.trees.add_collection("morphologies", self.storage)
        # self._nextId = 0
        # # Use the configuration to initialise all components such as cells and layers
        # # to prepare for the network architecture compilation.
        # self._intialise_components()
        # self._intialise_simulators()

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

    def _bootstrap(self, config, storage):
        # If both config and storage are given, overwrite the config in the storage. If
        # just the storage is given, load the config from storage. If neither is given,
        # create a default config and create a storage from it.
        if config is not None and storage is not None:
            self.storage = storage
            storage.store_config(config)
        elif storage is not None:
            config = storage.load_config()
        else:
            from bsb.storage import Storage

            if config is None:
                config = Configuration.default()
            storage = Storage(config.storage.engine, config.storage.root)

        self.configuration = config
        self.storage = storage
        self.storage.init(self)
        self.configuration._bootstrap(self)

    def resize(self, x=None, y=None, z=None):
        """
        Updates the topology boundary indicators. Use before placement, updates
        only the abstract topology tree, does not rescale, prune or otherwise
        alter already existing placement data.
        """
        from .topology import Boundary

        if x is not None:
            self.network.x = x
        if y is not None:
            self.network.y = y
        if z is not None:
            z = self.network.z
        self.topology.arrange(
            Boundary([0.0, 0.0, 0.0], [self.network.x, self.network.y, self.network.z])
        )

    def run_placement(self, strategies=None, DEBUG=True):
        """
        Run placement strategies.
        """
        if strategies is None:
            strategies = list(self.placement.values())
        strategies = PlacementStrategy.resolve_order(strategies)
        pool = create_job_pool(self)
        if pool.is_master():
            if strategies is None:
                types = self.get_cell_types()
                strategies = [c.placement for c in types]
            strategies = PlacementStrategy.resolve_order(strategies)
            for strategy in strategies:
                strategy.queue(pool, self.network.chunk_size)
            loop = self._progress_terminal_loop(pool, debug=DEBUG)
            try:
                pool.execute(loop)
            except:
                self._stop_progress_loop(loop, debug=DEBUG)
                raise
            finally:
                self._stop_progress_loop(loop, debug=DEBUG)
        else:
            pool.execute()

    def _progress_terminal_loop(self, pool, debug=False):
        import curses, time

        if debug:

            def loop(jobs):
                print("Total jobs:", len(jobs))
                print("Running jobs:", sum(1 for q in jobs if q._future.running()))
                print("Finished:", sum(1 for q in jobs if q._future.done()))
                time.sleep(1)

            return loop

        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)

        def loop(jobs):
            total = len(jobs)
            running = list(q for q in jobs if q._future.running())
            done = sum(1 for q in jobs if q._future.done())

            stdscr.clear()
            stdscr.addstr(0, 0, "-- Reconstruction progress --")
            stdscr.addstr(1, 2, f"Total jobs: {total}")
            stdscr.addstr(2, 2, f"Remaining jobs: {total - done}")
            stdscr.addstr(3, 2, f"Running jobs: {len(running)}")
            stdscr.addstr(4, 2, f"Finished jobs: {done}")
            for i, j in enumerate(running):
                stdscr.addstr(
                    6 + i,
                    2,
                    f"* Worker {i}: <{j._cname}>{j._name} {j._c}",
                )

            stdscr.refresh()
            time.sleep(0.1)

        loop._stdscr = stdscr
        return loop

    def _stop_progress_loop(self, loop, debug=False):
        if debug:
            return
        import curses

        curses.nocbreak()
        loop._stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    def run_placement_strategy(self, strategy):
        """
        Run a single placement strategy.
        """
        self.run_placement([strategy])

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

    def run_after_placement(self):
        """
        Run after placement hooks.
        """
        pool = create_job_pool(self)
        for hook in self.configuration.after_placement.values():
            pool.queue(hook.after_placement)
        pool.execute(self._pool_event_loop)

    def run_after_connectivity(self):
        """
        Run after placement hooks.
        """
        for hook in self.configuration.after_connectivity.values():
            hook.after_connectivity()

    def compile(self):
        """
        Run all steps in the scaffold sequence to obtain a full network.
        """
        t = time.time()
        self.run_placement()
        # self.run_after_placement()
        # self.run_connectivity()
        # self.run_after_connectivity()
        report("Runtime: {}".format(time.time() - t), 2)

    def clear(self):
        """
        Clears the storage. This deletes the network!
        """
        self.storage.renew(self)

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
        time_sim = time.time() - t
        report("Simulation runtime: {}".format(time_sim), level=2)
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

    def place_cells(self, cell_type, positions, morphologies=None, chunk=None):
        """
        Place cells inside of the scaffold

        .. code-block:: python

            # Add one granule cell at position 0, 0, 0
            cell_type = scaffold.get_cell_type("granule_cell")
            scaffold.place_cells(cell_type, cell_type.layer_instance, [[0., 0., 0.]])

        :param cell_type: The type of the cells to place.
        :type cell_type: :class:`.models.CellType`
        :param positions: A collection of xyz positions to place the cells on.
        :type positions: Any `np.concatenate` type of shape (N, 3).
        """
        if chunk is None:
            chunk = np.array([0, 0, 0])
        cell_count = positions.shape[0]
        if cell_count == 0:
            return
        self.get_placement_set(cell_type).append_data(chunk, positions, morphologies)

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
        :todo: Allow `additional` data for entities
        """
        if count == 0:
            return
        ps = self.get_placement_set(cell_type)
        # Append entity data to the default chunk 000
        ps.append_entities((0, 0, 0), count)

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

    def get_placement_of(self, *cell_types):
        """
        Find all of the placement strategies that include certain cell types.

        :param cell_types: Cell types of interest.
        :type cell_types: :class:`.objects.CellType`
        """

        def of(p):
            return any(ct in p.cell_types for ct in cell_types)

        return list(p for p in self.placement.values() if of(p))

    def get_placement_set(self, type):
        """
        Return a cell type's placement set from the output formatter.

        :param tag: Unique identifier of the placement set in the storage
        :type tag: string
        :returns: A placement set
        :rtype: :class:`.models.PlacementSet`
        """
        if isinstance(type, str):
            type = self.cell_types[type]
        return self.storage.get_placement_set(type)

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

        :param entities: In/exclude entity types
        :type entities: bool
        :returns: List of cell types
        """
        if entities:
            return list(self.configuration.cell_types.values())
        else:
            return [c for c in self.configuration.cell_types.values() if not c.entity]

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
        self.storage.Label(label).label(ids)

    def get_labels(self, pattern=None):
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
        if pattern is None:
            return self.storage._Label.list()
        if pattern.endswith("*"):
            p = pattern[:-1]
            finder = lambda l: l.startswith(p)
        else:
            finder = lambda l: l == pattern
        return list(filter(finder, self.storage._Label.list()))

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

    def create_filter(self, **kwargs):
        """
        Create a :class:`Filter <.storage.interfaces.Filter>`. Each keyword argument
        given to this function must match a supported filter type. The values of the
        keyword arguments are then set as a filter of that type.

        Filters need to be activated in order to exert their filtering function.
        """
        f = self.storage.create_filter(**kwargs)
        print("f:", f.__class__, dir(f))
        return self.storage.create_filter(**kwargs)

    def for_blender(self):
        """
        Binds all blender functions onto the scaffold object.
        """
        from .blender import _mixin

        for f_name, f in _mixin.__dict__.items():
            if callable(f) and not f_name.startswith("_"):
                self.__dict__[f_name] = f.__get__(self)

        return self

    def merge(self, other, label=None):
        warn(
            "The merge function currently only merges cell positions."
            + " Only cell types that exist in the calling network will be copied."
        )
        for ct in self.get_cell_types():
            if next((c for c in other.get_cell_types() if c.name == ct.name), None):
                ps = c.get_placement_set()
                ids = self.place_cells(ct, ct.layer_instance, ps.get_dataset())
                if label is not None:
                    self.label_cells(ids, label)
        self.compile_output()


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
