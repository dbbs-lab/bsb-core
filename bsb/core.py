import time
import os
import itertools
import typing
import numpy as np

from .placement import PlacementStrategy
from .connectivity import ConnectionStrategy
from .storage import Chunk, Storage, _util as _storutil, open_storage
from .exceptions import (
    InputError,
    NodeNotFoundError,
    RedoError,
    ScaffoldError,
)
from .reporting import report, warn
from .config._config import Configuration
from .services.pool import create_job_pool
from .services import MPI
from .simulation import get_simulation_adapter
from ._util import obj_str_insert
from .profiling import meter

if typing.TYPE_CHECKING:
    from .storage.interfaces import (
        PlacementSet,
        ConnectivitySet,
        MorphologyRepository,
        FileStore,
    )
    from .config._config import NetworkNode as Network
    from .simulation.simulation import Simulation
    from .topology import Region, Partition
    from .cell_types import CellType
    from .postprocessing import PostProcessingHook


@meter()
def from_storage(root):
    """
    Load :class:`.core.Scaffold` from a storage object.

    :param root: Root (usually path) pointing to the storage object.
    :returns: A network scaffold
    :rtype: :class:`Scaffold`
    """
    return open_storage(root).load()


_cfg_props = (
    "network",
    "regions",
    "partitions",
    "cell_types",
    "placement",
    "after_placement",
    "connectivity",
    "after_connectivity",
    "simulations",
)


def _config_property(name):
    def fget(self):
        return getattr(self.configuration, name)

    def fset(self, value):
        setattr(self.configuration, name, value)

    prop = property(fget)
    return prop.setter(fset)


def _get_linked_config(storage=None):
    import bsb.config

    try:
        cfg = storage.load_active_config()
    except Exception:
        import bsb.options

        path = bsb.options.config
    else:
        path = cfg._meta.get("path", None)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            cfg = bsb.config.from_file(f)
            return cfg
    else:
        return None


class Scaffold:
    """

    This is the main object of the bsb package, it represents a network and puts together
    all the pieces that make up the model description such as the
    :class:`~.config.Configuration` with the technical side like the
    :class:`~.storage.Storage`.
    """

    network: "Network"
    regions: typing.Mapping[str, "Region"]
    partitions: typing.Mapping[str, "Partition"]
    cell_types: typing.Mapping[str, "CellType"]
    placement: typing.Mapping[str, "PlacementStrategy"]
    after_placement: typing.Mapping[str, "PostProcessingHook"]
    connectivity: typing.Mapping[str, "ConnectionStrategy"]
    after_connectivity: typing.Mapping[str, "PostProcessingHook"]
    simulations: typing.Mapping[str, "Simulation"]

    def __init__(self, config=None, storage=None, clear=False, comm=None):
        """
        Bootstraps a network object.

        :param config: The configuration to use for this network. If it is omitted the
          :ref:`default configuration <default-config>` is used.
        :type config: :class:`~.config.Configuration`
        :param storage: The storage to use to read and write data for this network. If it
          is omitted the configuration's ``Storage`` node is used to construct one.
        :type storage: :class:`~.storage.Storage`
        :param clear: Start with a new network, clearing any previously stored information
        :type clear: bool
        :returns: A network object
        :rtype: :class:`~.core.Scaffold`
        """
        self._configuration = None
        self._storage = None
        self._comm = comm or MPI
        self._bootstrap(config, storage, clear=clear)

    def __contains__(self, component):
        return getattr(component, "scaffold", None) is self

    @obj_str_insert
    def __repr__(self):
        file = os.path.abspath(self.storage.root)
        cells_placed = len(self.cell_types)
        n_types = len(self.connectivity)
        return f"'{file}' with {cells_placed} cell types, and {n_types} connection_types"

    def is_main_process(self) -> bool:
        return not MPI.get_rank()

    def is_worker_process(self) -> bool:
        return bool(MPI.get_rank())

    def _bootstrap(self, config, storage, clear=False):
        if config is None:
            # No config given, check for linked configs, or stored configs, otherwise
            # make default config.
            linked = _get_linked_config(storage)
            if linked:
                report(f"Pulling configuration from linked {linked}.", level=2)
                config = linked
            elif storage is not None:
                config = storage.load_active_config()
            else:
                config = Configuration.default()
        if not storage:
            # No storage given, create one.
            report("Creating storage from config.", level=4)
            storage = Storage(config.storage.engine, config.storage.root)
        if clear:
            # Storage given, but asked to clear it before use.
            storage.remove()
            storage.create()
        # Synchronize the scaffold, config and storage objects for use together
        self._configuration = config
        # Make sure the storage config node reflects the storage we are using
        config._update_storage_node(storage)
        # Give the scaffold access to the unitialized storage object (for use during
        # config bootstrapping).
        self._storage = storage
        # First, the scaffold is passed to each config node, and their boot methods called
        self._configuration._bootstrap(self)
        # Then, `storage` is initted for the scaffold, and `config` is stored (happens
        # inside the `storage` property).
        self.storage = storage

    storage_cfg = _config_property("storage")
    for attr in _cfg_props:
        vars()[attr] = _config_property(attr)

    @property
    def configuration(self) -> Configuration:
        return self._configuration

    @configuration.setter
    def configuration(self, cfg: Configuration):
        self._configuration = cfg
        cfg._bootstrap(self)
        self.storage.store_active_config(cfg)

    @property
    def storage(self) -> Storage:
        return self._storage

    @storage.setter
    def storage(self, storage: Storage):
        self._storage = storage
        storage.init(self)

    @property
    def morphologies(self) -> "MorphologyRepository":
        return self.storage.morphologies

    @property
    def files(self) -> "FileStore":
        return self.storage.files

    def clear(self):
        """
        Clears the storage. This deletes any existing network data!
        """
        self.storage.renew(self)

    def clear_placement(self):
        """
        Clears the placement storage.
        """
        self.storage.clear_placement(self)

    def clear_connectivity(self):
        """
        Clears the connectivity storage.
        """
        self.storage.clear_connectivity()

    def resize(self, x=None, y=None, z=None):
        """
        Updates the topology boundary indicators. Use before placement, updates
        only the abstract topology tree, does not rescale, prune or otherwise
        alter already existing placement data.
        """
        from .topology._layout import box_layout

        if x is not None:
            self.network.x = x
        if y is not None:
            self.network.y = y
        if z is not None:
            z = self.network.z
        self.topology.do_layout(
            box_layout([0.0, 0.0, 0.0], [self.network.x, self.network.y, self.network.z])
        )

    @meter()
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
            except Exception:
                self._stop_progress_loop(loop, debug=DEBUG)
                raise
            finally:
                self._stop_progress_loop(loop, debug=DEBUG)
        else:
            pool.execute()

    @meter()
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
                strategy.queue(pool)
            loop = self._progress_terminal_loop(pool, debug=DEBUG)
            try:
                pool.execute(loop)
            except Exception:
                self._stop_progress_loop(loop, debug=DEBUG)
                raise
            finally:
                self._stop_progress_loop(loop, debug=DEBUG)
        else:
            pool.execute()

    @meter()
    def run_placement_strategy(self, strategy):
        """
        Run a single placement strategy.
        """
        self.run_placement([strategy])

    @meter()
    def run_after_placement(self):
        """
        Run after placement hooks.
        """
        if self.after_placement:
            warn("After placement disabled")
        # pool = create_job_pool(self)
        # for hook in self.configuration.after_placement.values():
        #     pool.queue(hook.after_placement)
        # pool.execute(self._pool_event_loop)

    @meter()
    def run_after_connectivity(self):
        """
        Run after placement hooks.
        """
        if self.after_connectivity:
            warn("After connectivity disabled")
        # for hook in self.configuration.after_connectivity.values():
        #     hook.after_connectivity()

    @meter()
    def compile(
        self,
        skip_placement=False,
        skip_connectivity=False,
        skip_after_placement=False,
        skip_after_connectivity=False,
        only=None,
        skip=None,
        clear=False,
        append=False,
        redo=False,
        force=False,
    ):
        """
        Run reconstruction steps in the scaffold sequence to obtain a full network.
        """
        existed = self.storage.preexisted
        p_strats = self.get_placement(skip=skip, only=only)
        c_strats = self.get_connectivity(skip=skip, only=only)
        todo_list_str = ", ".join(s.name for s in itertools.chain(p_strats, c_strats))
        report(f"Compiling the following strategies: {todo_list_str}", level=2)
        if sum((bool(clear), bool(redo), bool(append))) > 1:
            raise InputError("`clear`, `redo` and `append` are mutually exclusive.")
        if existed:
            if not (clear or append or redo):
                raise FileExistsError(
                    f"The `{self.storage.format}` storage"
                    + f" at `{self.storage.root}` already exists."
                    + " Use `clear`, `append` or `redo` to pick"
                    + " what to do with existing data."
                )
            if clear:
                report("Clearing data", level=2)
                self.clear_placement()
                self.clear_connectivity()
            elif redo:
                # In order to properly redo things, we clear some placement and connection
                # data, but since multiple placement/connection strategies can contribute
                # to the same sets we might be wiping their data too, and they will need
                # to be cleared and reran as well.
                p_strats, c_strats = self._redo_chain(p_strats, c_strats, skip, force)
            # else:
            #   append mode is luckily simpler, just don't clear anything :)

        t = time.time()
        if not skip_placement:
            placement_todo = ", ".join(s.name for s in p_strats)
            report(f"Starting placement strategies: {placement_todo}", level=3)
            self.run_placement(p_strats)
        if not skip_after_placement:
            self.run_after_placement()
        if not skip_connectivity:
            connectivity_todo = ", ".join(s.name for s in c_strats)
            report(f"Starting connectivity strategies: {connectivity_todo}", level=3)
            self.run_connectivity(c_strats)
        if not skip_after_connectivity:
            self.run_after_connectivity()
        report("Runtime: {}".format(time.time() - t), 2)
        # After compilation we should flag the storage as having existed before so that
        # the `clear`, `redo` and `append` flags take effect on a second `compile` pass.
        self.storage._preexisted = True

    @meter()
    def run_simulation(self, simulation_name: str, quit=False):
        """
        Run a simulation starting from the default single-instance adapter.

        :param simulation_name: Name of the simulation in the configuration.
        :type simulation_name: str
        """
        simulation = self.get_simulation(simulation_name)
        adapter = get_simulation_adapter(simulation.simulator)
        return adapter.simulate(simulation)

    def get_simulation(self, sim_name: str) -> "Simulation":
        """
        Retrieve the default single-instance adapter for a simulation.
        """
        if sim_name not in self.simulations:
            simstr = ", ".join(f"'{s}'" for s in self.simulations.keys())
            raise NodeNotFoundError(
                f"Unknown simulation '{sim_name}', choose from: {simstr}"
            )
        return self.configuration.simulations[sim_name]

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
        :type cell_type: ~bsb.cell_types.CellType
        :param positions: A collection of xyz positions to place the cells on.
        :type positions: Any `np.concatenate` type of shape (N, 3).
        """
        if chunk is None:
            chunk = Chunk([0, 0, 0], self.network.chunk_size)
        if hasattr(chunk, "dimensions") and np.any(np.isnan(chunk.dimensions)):
            chunk.dimensions = self.network.chunk_size
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
        :type cell_type: ~bsb.cell_types.CellType
        :param count: Number of entities to place
        :type count: int
        :todo: Allow `additional` data for entities
        """
        if count == 0:
            return
        ps = self.get_placement_set(cell_type)
        # Append entity data to the default chunk 000
        chunk = Chunk([0, 0, 0], self.network.chunk_size)
        ps.append_entities(chunk, count)

    def get_placement(
        self, cell_types=None, skip=None, only=None
    ) -> typing.List["PlacementStrategy"]:
        if cell_types is not None:
            cell_types = [
                self.cell_types[ct] if isinstance(ct, str) else ct for ct in cell_types
            ]
        return [
            val
            for key, val in self.placement.items()
            if (cell_types is None or any(ct in cell_types for ct in val.cell_types))
            and (only is None or key in only)
            and (skip is None or key not in skip)
        ]

    def get_placement_of(self, *cell_types):
        """
        Find all of the placement strategies that given certain cell types.

        :param cell_types: Cell types (or their names) of interest.
        :type cell_types: Union[~bsb.cell_types.CellType, str]
        """
        return self.get_placement(cell_types=cell_types)

    def get_placement_set(
        self, type, chunks=None, labels=None, morphology_labels=None
    ) -> "PlacementSet":
        """
        Return a cell type's placement set from the output formatter.

        :param tag: Unique identifier of the placement set in the storage
        :type tag: str
        :returns: A placement set
        :param labels: Labels to filter the placement set by.
        :type labels: list[str]
        :param morphology_labels: Subcellular labels to apply to the morphologies.
        :type morphology_labels: list[str]
        :rtype: :class:`~.storage.interfaces.PlacementSet`
        """
        if isinstance(type, str):
            type = self.cell_types[type]
        return self.storage.get_placement_set(
            type, chunks=chunks, labels=labels, morphology_labels=morphology_labels
        )

    def get_placement_sets(self) -> typing.List["PlacementSet"]:
        """
        Return all of the placement sets present in the network.

        :rtype: List[~bsb.storage.interfaces.PlacementSet]
        """
        return [cell_type.get_placement_set() for cell_type in self.cell_types.values()]

    def get_connectivity(
        self, anywhere=None, presynaptic=None, postsynaptic=None, skip=None, only=None
    ) -> typing.List["ConnectivitySet"]:
        conntype_filtered = self._connectivity_query(
            any_query=set(self._sanitize_ct(anywhere)),
            pre_query=set(self._sanitize_ct(presynaptic)),
            post_query=set(self._sanitize_ct(postsynaptic)),
        )
        return [
            ct
            for ct in conntype_filtered
            if (only is None or ct.name in only) and (skip is None or ct.name not in skip)
        ]

    def get_connectivity_sets(self) -> typing.List["ConnectivitySet"]:
        """
        Return all connectivity sets from the output formatter.

        :param tag: Unique identifier of the connectivity set in the output formatter
        :type tag: str
        :returns: All connectivity sets
        """
        return self.storage.get_connectivity_sets()

    def require_connectivity_set(self, pre, post, tag=None) -> "ConnectivitySet":
        return self.storage.require_connectivity_set(pre, post, tag)

    def get_connectivity_set(self, tag=None, pre=None, post=None) -> "ConnectivitySet":
        """
        Return a connectivity set from the output formatter.

        :param tag: Unique identifier of the connectivity set in the output formatter
        :type tag: str
        :returns: A connectivity set
        :rtype: :class:`~.storage.interfaces.ConnectivitySet`
        """
        if tag is None:
            try:
                tag = f"{pre.name}_to_{post.name}"
            except Exception:
                raise ValueError("Supply either `tag` or a valid pre and post cell type.")
        cs = self.storage.get_connectivity_set(tag)
        if pre and pre.name != cs._pre_name:
            raise ValueError(
                "Given and stored type mismatch:" + f" {pre.name} vs {cs._pre_name}"
            )
        if post and post.name != cs._post_name:
            raise ValueError(
                "Given and stored type mismatch:" + f" {post.name} vs {cs._post_name}"
            )
        try:
            cs.pre_type = self.cell_types[cs._pre_name]
            cs.post_type = self.cell_types[cs._post_name]
        except KeyError as e:
            raise NodeNotFoundError(f"Couldn't load {tag}, missing {e.args[0]}") from None
        return cs

    def get_cell_types(self) -> typing.List["CellType"]:
        """
        Return a list of all cell types in the network.
        """
        return [*self.configuration.cell_types.values()]

    def merge(self, other, label=None):
        raise NotImplementedError("Revisit: merge CT, PS & CS, done?")

    def _sanitize_ct(self, seq_str_or_none):
        if seq_str_or_none is None:
            return []
        try:
            if isinstance(seq_str_or_none, str):
                return [self.cell_types[seq_str_or_none]]
            return [
                self.cell_types[s] if isinstance(s, str) else s for s in seq_str_or_none
            ]
        except KeyError as e:
            raise NodeNotFoundError(f"Cell type `{e.args[0]}` not found.")

    def _progress_terminal_loop(self, pool, debug=False):
        import time

        if debug:

            def loop(jobs):
                print("Total jobs:", len(jobs))
                print("Running jobs:", sum(1 for q in jobs if q._future.running()))
                print("Finished:", sum(1 for q in jobs if q._future.done()))
                time.sleep(1)

            return loop

        import curses

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

    def _connectivity_query(self, any_query=set(), pre_query=set(), post_query=set()):
        # Filter network connection types for any type that satisfies both
        # the presynaptic and postsynaptic query. Empty queries satisfy all
        # types. The presynaptic query is satisfied if the conn type contains
        # any of the queried cell types presynaptically, and same for post.
        # The any query is satisfied if a cell type is found either pre or post.

        def partial_query(types, query):
            return not query or any(cell_type in query for cell_type in types)

        def query(conn_type):
            pre_match = partial_query(conn_type.presynaptic.cell_types, pre_query)
            post_match = partial_query(conn_type.postsynaptic.cell_types, post_query)
            any_match = partial_query(
                conn_type.presynaptic.cell_types, any_query
            ) or partial_query(conn_type.postsynaptic.cell_types, any_query)
            return any_match or (pre_match and post_match)

        types = self.connectivity.values()
        return [*filter(query, types)]

    def _redo_chain(self, p_strats, c_strats, skip, force):
        p_contrib = set(p_strats)
        while True:
            # Get all the placement strategies that effect the current set of CT.
            full_wipe = set(itertools.chain(*(ps.cell_types for ps in p_contrib)))
            contrib = set(self.get_placement(full_wipe))
            # Keep repeating until no new contributors are fished up.
            if contrib.issubset(p_contrib):
                break
            # Grow the placement chain
            p_contrib.update(contrib)
        report(
            "Redo-affected placement: " + " ".join(ps.name for ps in p_contrib), level=2
        )

        c_contrib = set(c_strats)
        conn_wipe = full_wipe.copy()
        while True:
            contrib = set(self.get_connectivity(anywhere=conn_wipe))
            conn_wipe.update(itertools.chain(*(ct.get_cell_types() for ct in contrib)))
            if contrib.issubset(c_contrib):
                break
            c_contrib.update(contrib)
        report(
            "Redo-affected connectivity: " + " ".join(cs.name for cs in c_contrib),
            level=2,
        )
        # Don't do greedy things without `force`
        if not force:
            # Error if we need to redo things the user asked to skip
            if skip is not None:
                unskipped = [p.name for p in p_contrib if p.name in skip]
                if unskipped:
                    chainstr = ", ".join(f"'{s.name}'" for s in (p_strats + c_strats))
                    skipstr = ", ".join(f"'{s.name}'" for s in unskipped)
                    raise RedoError(
                        f"Can't skip {skipstr}. Redoing {chainstr} requires to redo them."
                        + f" Omit {skipstr} from `skip` or use `force` (not recommended)."
                    )
            # Error if we need to redo things the user didn't ask for
            for label, chain, og in zip(
                ("placement", "connection"), (p_contrib, c_contrib), (p_strats, c_strats)
            ):
                if len(chain) > len(og):
                    new = chain.difference(og)
                    raise RedoError(
                        f"Need to redo additional {label} strategies: "
                        + ", ".join(n.name for n in new)
                        + ". Include them or use `force` (not recommended)."
                    )

        for ct in full_wipe:
            report(f"Clearing all data of {ct.name}", level=2)
            ct.clear()

        for ct in conn_wipe:
            report(f"Clearing connectivity data of {ct.name}", level=2)
            ct.clear_connections()

        return p_contrib, c_contrib


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
