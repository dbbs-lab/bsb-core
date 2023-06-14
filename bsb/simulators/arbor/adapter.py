import itertools
import typing

from bsb.reporting import report, warn
from bsb.exceptions import AdapterError, UnknownGIDError
from bsb.services import MPI
from bsb.simulation.adapter import SimulatorAdapter
import numpy as np
import itertools as it
import time
import arbor

from ...simulation.results import SimulationResult
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
        endpoint = str(rcv.loc_on)
        id = self._endpoint_counters.get(endpoint, 0)
        self._endpoint_counters[endpoint] = id + 1
        rcv.index = id
        super().append(rcv)


class QuickContains:
    def __init__(self, simdata, cell_model, offset):
        self._model = cell_model
        ps = cell_model.get_placement_set(simdata.chunks)
        self._ranges = self._get_ranges(simdata.chunks, ps, offset)

    @property
    def model(self):
        return self._model

    def __contains__(self, i):
        return any(start <= i < stop for start, stop in self._ranges)

    def _get_ranges(self, chunks, ps, offset):
        stats = ps.get_chunk_stats()
        ranges = []
        for chunk, len_ in sorted(
            stats.items(), key=lambda k: Chunk.from_id(int(k[0]), None).id
        ):
            if chunk in chunks:
                ranges.append((offset, offset + len_))
            offset += len_
        return ranges

    def __iter__(self):
        yield from itertools.chain.from_iterable(range(r[0], r[1]) for r in self._ranges)


class GIDManager:
    def __init__(self, simulation, simdata):
        self._gid_offsets = {}
        self._model_order = sorted(
            simulation.cell_models.values(),
            key=lambda model: len(model.get_placement_set()),
        )
        ctr = 0
        for model in self._model_order:
            self._gid_offsets[model] = ctr
            ctr += len(model.get_placement_set())
        self._contains = [
            QuickContains(simdata, model, offset)
            for model, offset in self._gid_offsets.items()
        ]

    def lookup_kind(self, gid):
        return self._lookup(gid).model.get_cell_kind(gid)

    def lookup_model(self, gid):
        return self._lookup(gid).model

    def _lookup(self, gid):
        try:
            return next(c for c in self._contains if gid in c)
        except StopIteration:
            raise UnknownGIDError(f"Can't find gid {gid}.") from None

    def all(self):
        yield from itertools.chain.from_iterable(self._contains)


class ArborRecipe(arbor.recipe):
    def __init__(self, simulation, simdata):
        super().__init__()
        self._simulation = simulation
        self._simdata = simdata
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
        self._global_properties.catalogue = self._get_catalogue()

    def _get_catalogue(self):
        catalogue = arbor.default_catalogue()
        prefixes = set()
        for model in self._simulation.cell_models.values():
            prefix, model_catalogue = model.get_prefixed_catalogue()
            if model_catalogue is not None and prefix not in prefixes:
                prefixes.add(prefix)
                catalogue.extend(model_catalogue, "")

        return catalogue

    def global_properties(self, kind):
        return self._global_properties

    def num_cells(self):
        return sum(
            len(model.get_placement_set())
            for model in self._simulation.cell_models.values()
        )

    def cell_kind(self, gid):
        return self._simdata._lookup.lookup_kind(gid)

    def cell_description(self, gid):
        model = self._adapter._lookup.lookup_model(gid)
        return model.get_description(gid)

    def connections_on(self, gid):
        return [
            arbor.connection(rcv.from_(), rcv.on(), rcv.weight, rcv.delay)
            for rcv in self._adapter._connections_on[gid]
        ]

    def gap_junctions_on(self, gid):
        return [c.model.gap_(c) for c in self._simdata.gap_junctions_on.get(gid, [])]

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


class ConnectionWrapper:
    def __init__(self, pre_loc, post_loc):
        self.from_id = pre_loc[0]
        self.to_id = post_loc[0]
        self.pre_loc = pre_loc[1:]
        self.post_loc = post_loc[1:]


class ArborAdapter(SimulatorAdapter):
    def __init__(self):
        super().__init__()
        self.simdata = {}

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
                context = arbor.context(arbor.proc_allocation(simulation.threads))
            if simulation.profiling:
                if arbor.config()["profiling"]:
                    report("enabling profiler", level=2)
                    arbor.profiler_initialize(context)
                else:
                    raise RuntimeError(
                        "Arbor must be built with profiling support to use the `profiling` flag."
                    )
            simdata.gid_manager = self.get_gid_manager(simulation, simdata)
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

    def get_gid_manager(self, simulation, simdata):
        return GIDManager(simulation, simdata)

    def prepare_samples(self, sim):
        for device in self.devices.values():
            device.prepare_samples(sim)

    def run(self, simulation):
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
        self._cache_gap_junctions(simulation, simdata)
        self._cache_connections(simulation, simdata)
        self._cache_devices(simulation, simdata)
        return ArborRecipe(simulation, simdata)

    def _create_simdata(self, simulation):
        self.simdata[simulation] = simdata = SimulationData(simulation)
        self._assign_chunks(simulation, simdata)
        return simdata

    def _cache_gap_junctions(self, simulation, simdata):
        simdata.gap_junctions_on = {}
        for conn_model in simulation.connection_models.values():
            if conn_model.gap:
                conn_set = conn_model.get_connectivity_set()
                conns = conn_set.load_connections().to(simdata.chunks).as_globals()
                conn_model.create_gap_junctions_on(simdata.gap_junctions_on, conns)

    def _cache_connections(self, simulation, simdata):
        simdata.connections_on = {
            gid: ReceiverCollection() for gid in simdata.gid_manager.all()
        }
        simdata.connections_from = {gid: [] for gid in simdata.gid_manager.all()}
        for conn_model in simulation.connection_models.values():
            if conn_model.gap:
                continue
            conn_set = conn_model.get_connectivity_set()
            # Load the arriving connection iterator
            conns_on = conn_set.load_connections().to(simdata.chunks).as_globals()
            # Create the arriving connections
            conn_model.create_connections_on(simdata.connections_on, conns_on)
            # Load the outgoing connection iterator
            conns_from = conn_set.load_connections().from_(simdata.chunks).as_globals()
            # Create the outgoing connections
            conn_model.create_connections_from(simdata.connections_from, conns_from)

    def _cache_devices(self, simulation, simdata):
        simdata.devices_on = {gid: [] for gid in simdata.gid_manager.all()}
        for device in simulation.devices.values():
            targets = device.get_targets(self)
            for target in targets:
                simdata.devices_on[target].append(device)

    def _assign_chunks(self, simulation, simdata):
        chunk_stats = simulation.scaffold.storage.get_chunk_stats()
        size = MPI.get_size()
        all_chunks = [Chunk.from_id(int(chunk), None) for chunk in chunk_stats.keys()]
        simdata.node_chunk_alloc = [all_chunks[rank::size] for rank in range(0, size)]
        simdata.chunk_node_map = {}
        for node, chunks in enumerate(simdata.node_chunk_alloc):
            for chunk in chunks:
                simdata.chunk_node_map[chunk] = node
        simdata.chunks = simdata.node_chunk_alloc[MPI.get_rank()]
