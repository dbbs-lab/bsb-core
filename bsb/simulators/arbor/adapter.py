import typing

from bsb.reporting import report, warn
from bsb.exceptions import AdapterError
from bsb.services import MPI
from bsb.simulation.adapter import SimulatorAdapter
import numpy as np
import itertools as it
import time
import arbor

from ...storage import Chunk

if typing.TYPE_CHECKING:
    from .simulation import ArborSimulation


class SimulationData:
    def __init__(self, simulation):
        self.chunks = None
        self.populations = dict()
        self.placement = {
            model: model.get_placement_set() for model in simulation.cell_models.values()
        }
        self.connections = dict()
        self.devices = dict()
        self.result: "NestResult" = None


class ReceiverCollection(list):
    def __init__(self):
        super().__init__()
        self._endpoint_counters = {}

    def append(self, rcv):
        endpoint = rcv.comp_on.id
        id = self._endpoint_counters.get(endpoint, 0)
        self._endpoint_counters[endpoint] = id + 1
        rcv.index = id
        super().append(rcv)

    @property
    def weight(self):
        return self.conn_model.weight

    @property
    def delay(self):
        return self.conn_model.delay

    def from_(self):
        return arbor.cell_global_label(self.from_gid, f"comp_{self.comp_from.id}")

    def on(self):
        return arbor.cell_local_label(f"comp_{self.comp_on.id}_{self.index}")


class QuickContains:
    def __init__(self, cell_model, ps):
        self._model = cell_model
        self._ps = ps
        self._type = ps.type
        if cell_model.relay or ps.type.entity:
            self._kind = arbor.cell_kind.spike_source
        else:
            self._kind = arbor.cell_kind.cable
        self._ranges = [
            (start, start + count)
            for start, count in continuity_hop(iter(ps._identifiers.get_dataset()))
        ]

    def __contains__(self, i):
        return any(i >= start and i < stop for start, stop in self._ranges)


class QuickLookup:
    def __init__(self, simulation):
        self._contains = [
            QuickContains(model, model.get_placement_set(model.name))
            for model in simulation.cell_models.values()
        ]

    def lookup_kind(self, gid):
        return self._lookup(gid)._kind

    def lookup_model(self, gid):
        return self._lookup(gid)._model

    def _lookup(self, gid):
        try:
            return next(c for c in self._contains if gid in c)
        except StopIteration:
            raise UnknownGIDError(f"Can't find gid {gid}.") from None


class ArborRecipe(arbor.recipe):
    def __init__(self, simulation, simdata):
        super().__init__()
        self._simulation = simulation
        self._simdata = simdata
        self._catalogue = self._get_catalogue()
        self._global_properties = arbor.neuron_cable_properties()
        self._global_properties.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self._global_properties.set_ion(ion="na", int_con=10, ext_con=140, rev_pot=50)
        self._global_properties.set_ion(ion="k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        self._global_properties.set_ion(
            ion="ca", int_con=0.0001, ext_con=2, rev_pot=132.5
        )
        self._global_properties.set_ion(
            ion="h", valence=1, int_con=1.0, ext_con=1.0, rev_pot=-34
        )
        self._global_properties.register(self._catalogue)

    def _get_catalogue(self):
        catalogue = arbor.default_catalogue()
        prefixes = set()
        for cell in self._adapter.cell_models.values():
            # Add the unique set of catalogues of non relay models to the recipe
            # catalogue.
            if (
                cell.model_class
                and (p := cell.model_class.get_catalogue_prefix()) not in prefixes
            ):
                prefixes.add(p)
                catalogue.extend(cell.model_class.get_catalogue(), "")

        return catalogue

    def global_properties(self, kind):
        return self._global_properties

    def num_cells(self):
        return sum(
            len(model.get_placement_set())
            for model in self._simulation.cell_models.values()
        )

    def cell_kind(self, gid):
        return self._adapter._lookup.lookup_kind(gid)

    def cell_description(self, gid):
        model = self._adapter._lookup.lookup_model(gid)
        return model.get_description(gid)

    def connections_on(self, gid):
        return [
            arbor.connection(rcv.from_(), rcv.on(), rcv.weight, rcv.delay)
            for rcv in self._adapter._connections_on[gid]
        ]

    def gap_junctions_on(self, gid):
        return [c.model.gap_(c) for c in self._adapter._gap_junctions_on.get(gid, [])]

    def probes(self, gid):
        devices = self._adapter._devices_on[gid]
        _ntag = 0
        probes = []
        for device in devices:
            device_probes = device.implement(gid)
            for tag in range(_ntag, _ntag + len(device_probes)):
                device.register_probe_id(gid, tag)
            probes.extend(device_probes)
        return probes

    def _name_of(self, gid):
        return self._adapter._lookup._lookup(gid)._type.name


class ArborAdapter(SimulatorAdapter):
    def get_rank(self):
        return MPI.get_rank()

    def get_size(self):
        return MPI.get_size()

    def broadcast(self, data, root=0):
        return MPI.bcast(data, root)

    def barrier(self):
        return MPI.barrier()

    def prepare(self, simulation: "ArborSimulation", comm=None):
        simdata = self._create_simdata(simulation)
        try:
            try:
                if hasattr(MPI, "_mocked"):
                    mpi = None
                else:
                    mpi = MPI
                context = arbor.context(
                    arbor.proc_allocation(simulation.threads), comm=mpi
                )
            except TypeError:
                if MPI.get_size() > 1:
                    s = MPI.get_size()
                    warn(
                        f"Arbor does not seem to be built with MPI support, running duplicate simulations on {s} nodes."
                    )
                context = arbor.context(arbor.proc_allocation(self.threads))
            if simulation.profiling:
                if arbor.config()["profiling"]:
                    report("enabling profiler", level=2)
                    arbor.profiler_initialize(context)
                else:
                    raise RuntimeError(
                        "Arbor must be built with profiling support to use the `profiling` flag."
                    )
            simdata.lookup = QuickLookup(simulation)
            report("preparing simulation", level=1)
            report("MPI processes:", context.ranks, level=2)
            report("Threads per process:", context.threads, level=2)
            recipe = self.get_recipe(simulation, simdata)
            # Gap junctions are required for domain decomposition
            self.domain = arbor.partition_load_balance(recipe, context)
            self.gids = set(it.chain.from_iterable(g.gids for g in self.domain.groups))
            simulation = arbor.simulation(recipe, self.domain, context)
            self.prepare_samples(simulation)
            report("prepared simulation", level=1)
            return simulation
        except Exception:
            del self.simdata[simulation]
            raise

    def prepare_samples(self, sim):
        for device in self.devices.values():
            device.prepare_samples(sim)

    def simulate(self, simulation):
        if not MPI.get_rank():
            simulation.record(arbor.spike_recording.all)
        start = time.time()
        report("running simulation", level=1)
        self.start_progress(self.duration)
        for oi, i in self.step_progress(self.duration, 1):
            simulation.run(i, dt=self.resolution)
            self.progress(i)
        report(f"completed simulation. {time.time() - start:.2f}s", level=1)
        if self.profiling and arbor.config()["profiling"]:
            report("printing profiler summary", level=2)
            report(arbor.profiler_summary(), level=1)

    def collect_output(self, simulation):
        import h5py, time, random, traceback

        timestamp = str(time.time()).split(".")[0] + str(random.random()).split(".")[1]
        timestamp = self.broadcast(timestamp)
        result_path = "results_" + self.name + "_" + timestamp + ".hdf5"
        rank = self.get_rank()
        for node in range(self.get_size()):
            self.barrier()
            if node == rank:
                report("Node", rank, "is writing", level=2, all_nodes=True)
                with h5py.File(result_path, "a") as f:
                    if rank == 0:
                        spikes = simulation.spikes()
                        spikes = np.column_stack(
                            (
                                np.fromiter((l[0][0] for l in spikes), dtype=int),
                                np.fromiter((l[1] for l in spikes), dtype=int),
                            )
                        )
                        f.create_dataset("all_spikes_dump", data=spikes)
                    f.attrs["configuration_string"] = self.scaffold.configuration._raw
                    for path, data, meta in self.result.safe_collect():
                        try:
                            path = "/".join(f"{p}" for p in path)
                            if path in f:
                                # Path exists, append by recreating concatenated data
                                data = np.concatenate((f[path][()], data))
                                _meta = d.attrs[k].copy()
                                _meta.update(meta)
                                meta = _meta
                                del f[path]
                            d = f.create_dataset(path, data=data)
                            for k, v in meta.items():
                                d.attrs[k] = v
                        except Exception as e:
                            if not isinstance(data, np.ndarray):
                                warn(
                                    f"Recorder `{path}` expected numpy.ndarray data,"
                                    + f" got {type(data)}"
                                )
                            else:
                                warn(
                                    f"Recorder {path} processing errored out."
                                    + f" - Data: {data.dtype} {data.shape}"
                                    + f"\n\n{traceback.format_exc()}"
                                )
            self.barrier()
        return result_path

    def get_recipe(self, simulation, simdata=None):
        if simdata is None:
            simdata = self._create_simdata(simulation)
        self._cache_gap_junctions()
        self._cache_connections()
        self.prepare_devices()
        self._cache_devices()
        return ArborRecipe(simulation, simdata)

    def _create_simdata(self, simulation):
        self.simdata[simulation] = simdata = SimulationData(simulation)
        self._assign_chunks(simulation)
        return simdata

    def _cache_gap_junctions(self):
        self._gap_junctions_on = {}
        for conn_set in self.scaffold.get_connectivity_sets():
            if conn_set.is_orphan() or not len(conn_set):
                continue
            try:
                conn_model = self.connection_models[conn_set.tag]
            except KeyError:
                raise AdapterError(f"Missing connection model `{conn_set.tag}`")
            if not conn_model.gap:
                continue
            for conn in conn_set.intersections:
                conn.model = conn_model
                self._gap_junctions_on.setdefault(conn.from_id, []).append(conn)

    def _cache_connections(self):
        self._connections_on = {gid: ReceiverCollection() for gid in self.gids}
        self._connections_from = {gid: [] for gid in self.gids}
        for conn_set in self.scaffold.get_connectivity_sets():
            if conn_set.is_orphan() or not len(conn_set):
                continue
            ct = conn_set.connection_types[0]
            try:
                conn_model = self.connection_models[conn_set.tag]
            except KeyError:
                raise AdapterError(f"Missing connection model `{conn_set.tag}`")
            if conn_model.gap:
                continue
            w = conn_model.weight
            for conn in conn_set.intersections:
                from_gid = int(conn.from_id)
                to_gid = int(conn.to_id)
                comp_from = conn.from_compartment
                comp_on = conn.to_compartment
                if from_gid in self._connections_from:
                    self._connections_from[from_gid].append(comp_from)
                if to_gid in self._connections_on:
                    self._connections_on[to_gid].append(
                        conn_model.make_receiver(from_gid, comp_from, comp_on)
                    )
            for gid, relays in self._relays_on.items():
                for from_gid, comp_from, comp_on, conn_model in relays:
                    self._connections_from[from_gid].append(comp_from)
                    self._connections_on[gid].append(
                        conn_model.make_receiver(from_gid, comp_from, comp_on)
                    )

    def prepare_devices(self):
        device_module = __import__("devices", globals(), level=1).__dict__
        for device in self.devices.values():
            device_class = "".join(x.title() for x in device.device.split("_"))
            device._bootstrap(device_module[device_class])

    def _cache_devices(self):
        self._devices_on = {gid: [] for gid in self.gids}
        for device in self.devices.values():
            targets = device.get_targets(self)
            for target in targets:
                self._devices_on[target].append(device)

    def _assign_chunks(self, simulation):
        simdata = self.simdata[simulation]
        chunk_stats = simulation.scaffold.storage.get_chunk_stats()
        size = MPI.get_size()
        all_chunks = [Chunk.from_id(int(chunk), None) for chunk in chunk_stats.keys()]
        simdata.node_chunk_alloc = [all_chunks[rank::size] for rank in range(0, size)]
        simdata.chunk_node_map = {}
        for node, chunks in enumerate(simdata.node_chunk_alloc):
            for chunk in chunks:
                simdata.chunk_node_map[chunk] = node
        simdata.chunks = simdata.node_chunk_alloc[MPI.get_rank()]
