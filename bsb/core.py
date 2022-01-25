from .statistics import Statistics
from .plotting import plot_network
import numpy as np
import time
from .helpers import map_ndarray, listify_input
from .placement import PlacementStrategy
from .connectivity import ConnectionStrategy
from warnings import warn as std_warn
from .exceptions import *
from .reporting import report, warn, has_mpi_installed, get_report_file
from .config._config import Configuration
from ._pool import create_job_pool


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

    def clear(self):
        """
        Clears the storage. This deletes any existing network data!
        """
        self.storage.renew(self)

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

    def run_connectivity(self, strategies=None, DEBUG=True):
        """
        Run connection strategies.
        """
        if strategies is None:
            strategies = list(self.connectivity.values())
        strategies = ConnectionStrategy.resolve_order(strategies)
        pool = create_job_pool(self)
        if pool.is_master():
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

    def run_after_placement(self):
        """
        Run after placement hooks.
        """
        warn("After placement disabled")
        # pool = create_job_pool(self)
        # for hook in self.configuration.after_placement.values():
        #     pool.queue(hook.after_placement)
        # pool.execute(self._pool_event_loop)

    def run_after_connectivity(self):
        """
        Run after placement hooks.
        """
        warn("After connectivity disabled")
        # for hook in self.configuration.after_connectivity.values():
        #     hook.after_connectivity()

    def compile(
        self,
        skip_placement=False,
        skip_connectivity=False,
        skip_after_placement=False,
        skip_after_connectivity=False,
    ):
        """
        Run all steps in the scaffold sequence to obtain a full network.
        """
        t = time.time()
        if not skip_placement:
            self.run_placement()
        if not skip_after_placement:
            self.run_after_placement()
        if not skip_connectivity:
            self.run_connectivity()
        if not skip_after_connectivity:
            self.run_after_connectivity()
        report("Runtime: {}".format(time.time() - t), 2)

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

    def place_cells(
        self,
        cell_type,
        positions,
        morphologies=None,
        rotations=None,
        additional=None,
        chunk=None,
    ):
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
        self.get_placement_set(cell_type).append_data(
            chunk,
            positions=positions,
            morphologies=morphologies,
            rotations=rotations,
            additional=additional,
        )

    def connect_cells(self):
        raise NotImplementedError("hehe, todo!")

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

    def get_placement_set(self, type, chunks=None):
        """
        Return a cell type's placement set from the output formatter.

        :param tag: Unique identifier of the placement set in the storage
        :type tag: string
        :returns: A placement set
        :rtype: :class:`.models.PlacementSet`
        """
        if isinstance(type, str):
            type = self.cell_types[type]
        return self.storage.get_placement_set(type, chunks=chunks)

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
