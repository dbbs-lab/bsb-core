from ...simulation import (
    Simulation,
    CellModel,
    ConnectionModel,
    DeviceModel,
    SimulationResult,
    SimulationRecorder,
)
from ...simulation.targetting import NeuronTargetting
from ... import config
from ...config import types
from ...reporting import report, warn
from ...exceptions import *
from ...services import MPI
import numpy as np
import itertools as it
import functools
import os
import time
import psutil
import collections

try:
    import arbor

    _has_arbor = True
except ImportError:
    _has_arbor = False
    import types as _t

    # Mock missing requirements, as arbor is, like
    # all simulators, an optional dep. of the BSB.
    arbor = _t.ModuleType("arbor")
    arbor.recipe = type("mock_recipe", (), dict())

    def get(*arg):
        raise ImportError("Arbor not installed.")

    arbor.__getattr__ = get


def _consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


it.consume = _consume


class ArborCell(CellModel):
    node_name = "simulations.?.cell_models"
    default_endpoint = "comp_-1"

    def validate(self):
        self.model_class = None
        if _has_arbor and not self.relay:
            self.model_class = get_configurable_class(self.model)

    def get_description(self, gid):
        if not self.relay:
            morphology, labels, decor = self.model_class.cable_cell_template()
            labels = self._add_labels(gid, labels, morphology)
            decor = self._add_decor(gid, decor)
            cc = arbor.cable_cell(morphology, labels, decor)
            return cc
        else:
            schedule = self.get_schedule(gid)
            return arbor.spike_source_cell(self.default_endpoint, schedule)

    def get_schedule(self, gid):
        schedule = arbor.explicit_schedule([])
        for device in self.adapter._devices_on[gid]:
            pattern = device.get_pattern(gid)
            if not pattern:
                continue
            merged = pattern + schedule.events(0, float("inf"))
            schedule = arbor.explicit_schedule(merged)
        return schedule

    def _add_decor(self, gid, decor):
        self._soma_detector(decor)
        self._create_transmitters(gid, decor)
        self._create_gaps(gid, decor)
        self._create_receivers(gid, decor)
        return decor

    def _add_labels(self, gid, labels, morphology):
        pwlin = arbor.place_pwlin(morphology)

        def comp_label(comp):
            if comp.id == -1:
                warn(f"Encountered nil compartment on {gid}")
                return
            loc, d = pwlin.closest(*comp.start)
            if d > 0.0001:
                raise AdapterError(f"Couldn't find {comp.start}, on {self._str(gid)}")
            labels[f"comp_{comp.id}"] = str(loc)

        comps_from = self.adapter._connections_from[gid]
        comps_on = (rcv.comp_on for rcv in self.adapter._connections_on[gid])
        gaps = (c.to_compartment for c in self.adapter._gap_junctions_on.get(gid, []))
        it.consume(comp_label(i) for i in it.chain(comps_from, comps_on, gaps))
        labels[self.default_endpoint] = "(root)"
        return labels

    def _str(self, gid):
        return f"{self.adapter._name_of(gid)} {gid}"

    def _soma_detector(self, decor):
        decor.place("(root)", arbor.spike_detector(-10), self.default_endpoint)

    def _create_transmitters(self, gid, decor):
        done = set()
        for comp in self.adapter._connections_from[gid]:
            if comp.id in done:
                continue
            else:
                done.add(comp.id)
            decor.place(f'"comp_{comp.id}"', arbor.spike_detector(-10), f"comp_{comp.id}")

    def _create_gaps(self, gid, decor):
        done = set()
        for conn in self.adapter._gap_junctions_on.get(gid, []):
            comp = conn.to_compartment
            if comp.id in done:
                continue
            else:
                done.add(comp.id)
            decor.place(f'"comp_{comp.id}"', arbor.junction("gj"), f"gap_{comp.id}")

    def _create_receivers(self, gid, decor):
        for rcv in self.adapter._connections_on[gid]:
            decor.place(
                f'"comp_{rcv.comp_on.id}"',
                rcv.synapse,
                f"comp_{rcv.comp_on.id}_{rcv.index}",
            )


@config.node
class ArborDevice(DeviceModel):
    targetting = config.attr(type=NeuronTargetting, required=True)
    resolution = config.attr(type=float)
    sampling_policy = config.attr(type=types.in_([""]))

    defaults = {"resolution": None, "sampling_policy": "exact"}

    def __boot__(self):
        self.resolution = self.resolution or self.adapter.resolution

    def register_probe_id(self, gid, tag):
        self._probe_ids.append((gid, tag))

    def prepare_samples(self, sim):
        self._handles = [self.sample(sim, probe_id) for probe_id in self._probe_ids]

    def sample(self, sim, probe_id):
        schedule = arbor.regular_schedule(self.resolution)
        sampling_policy = getattr(arbor.sampling_policy, self.sampling_policy)
        return sim.sample(probe_id, schedule, sampling_policy)

    def get_samples(self, sim):
        return [sim.samples(handle) for handle in self._handles]

    def get_meta(self):
        attrs = ("name", "sampling_policy", "resolution")
        return dict(zip(attrs, (getattr(self, attr) for attr in attrs)))


class ArborConnection(ConnectionModel):
    defaults = {"gap": False, "delay": 0.025, "weight": 1.0}
    casts = {"delay": float, "gap": bool, "weight": float}

    def validate(self):
        pass

    def make_receiver(*args):
        return Receiver(*args)

    def gap_(self, conn):
        l = arbor.cell_local_label(f"gap_{conn.to_compartment.id}")
        g = arbor.cell_global_label(int(conn.from_id), f"gap_{conn.from_compartment.id}")
        return arbor.gap_junction_connection(g, l, self.weight)


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


class Receiver:
    def __init__(self, conn_model, from_gid, comp_from, comp_on):
        self.conn_model = conn_model
        self.from_gid = from_gid
        self.comp_from = comp_from
        self.comp_on = comp_on
        self.synapse = arbor.synapse("expsyn")

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
    def __init__(self, adapter):
        network = adapter.scaffold
        self._contains = [
            QuickContains(model, network.get_placement_set(model.name))
            for model in adapter.cell_models.values()
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
    def __init__(self, adapter):
        super().__init__()
        self._adapter = adapter
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
        adapter = self._adapter
        network = adapter.scaffold
        s = sum(len(ps) for ps in map(network.get_placement_set, adapter.cell_models))
        return s

    def num_sources(self, gid):
        return 1 if self._is_relay(gid) else 0

    def cell_kind(self, gid):
        return self._adapter._lookup.lookup_kind(gid)

    def cell_description(self, gid):
        model = self._adapter._lookup.lookup_model(gid)
        return model.get_description(gid)

    def connections_on(self, gid):
        if self._is_relay(gid):
            return []
        return [
            arbor.connection(rcv.from_(), rcv.on(), rcv.weight, rcv.delay)
            for rcv in self._adapter._connections_on[gid]
        ]

    def gap_junctions_on(self, gid):
        return [c.model.gap_(c) for c in self._adapter._gap_junctions_on.get(gid, [])]

    def _is_relay(self, gid):
        return self._adapter._lookup.lookup_kind(gid) == arbor.cell_kind.spike_source

    def probes(self, gid):
        if self._is_relay(gid):
            return []
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


@config.node
class ArborSimulation(Simulation):
    duration = config.attr(type=float, required=True)

    defaults = {"threads": 1, "profiling": True, "resolution": 0.025}

    def validate(self):
        if self.threads == "all":
            self.threads = psutil.cpu_count(logical=False)

    def get_rank(self):
        return MPI.get_rank()

    def get_size(self):
        return MPI.get_size()

    def broadcast(self, data, root=0):
        return MPI.bcast(data, root)

    def barrier(self):
        return MPI.barrier()

    def init_result(self):
        self.result = SimulationResult()

    def prepare(self):
        try:
            self.scaffold.assert_continuity()
        except AssertionError as e:
            raise AssertionError(
                str(e) + " The arbor adapter requires completely continuous GIDs."
            ) from None
        try:
            if hasattr(MPI, "_mocked"):
                mpi = None
            else:
                mpi = MPI
            context = arbor.context(arbor.proc_allocation(self.threads), comm=mpi)
        except TypeError:
            if MPI.get_size() > 1:
                s = MPI.get_size()
                warn(
                    f"Arbor does not seem to be built with MPI support, running duplicate simulations on {s} nodes."
                )
            context = arbor.context(arbor.proc_allocation(self.threads))
        if self.profiling and arbor.config()["profiling"]:
            report("enabling profiler", level=2)
            arbor.profiler_initialize(context)
        self.init_result()
        self._lookup = QuickLookup(self)
        report("preparing simulation", level=1)
        report("MPI processes:", context.ranks, level=2)
        report("Threads per process:", context.threads, level=2)
        recipe = self.get_recipe()
        # Gap junctions are required for domain decomposition
        self._cache_gap_junctions()
        self.domain = arbor.partition_load_balance(recipe, context)
        self.gids = set(it.chain.from_iterable(g.gids for g in self.domain.groups))
        # Cache uses the domain decomposition to cache info per gid on this node. The
        # recipe functions use the cache, but luckily aren't called until
        # `arbor.simulation` and `simulation.run`.
        self._index_relays()
        self._cache_connections()
        self.prepare_devices()
        self._cache_devices()
        simulation = arbor.simulation(recipe, self.domain, context)
        self.prepare_samples(simulation)
        report("prepared simulation", level=1)
        return simulation

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

    def get_recipe(self):
        return ArborRecipe(self)

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
                for (from_gid, comp_from, comp_on, conn_model) in relays:
                    self._connections_from[from_gid].append(comp_from)
                    self._connections_on[gid].append(
                        conn_model.make_receiver(from_gid, comp_from, comp_on)
                    )

    def _index_relays(self):
        report("Indexing relays.")
        terminal_relays = {}
        intermediate_relays = {}
        output_handler = self.scaffold.output_formatter
        cell_types = self.scaffold.get_cell_types()
        type_lookup = {
            ct.name: range(min(ids), max(ids) + 1)
            for ct, ids in zip(
                cell_types, (ct.get_placement_set().identifiers for ct in cell_types)
            )
        }

        def lookup(i):
            for n, t in type_lookup.items():
                if i in t:
                    return n
            else:
                return None

        for connection_model in self.connection_models.values():
            name = connection_model.name
            # Get the connectivity set associated with this connection model
            connectivity_set = ConnectivitySet(output_handler, connection_model.name)
            if connectivity_set.is_orphan():
                continue
            from_cell_type = connectivity_set.connection_types[0].from_cell_types[0]
            from_cell_model = self.cell_models[from_cell_type.name]
            to_cell_type = connectivity_set.connection_types[0].to_cell_types[0]
            to_cell_model = self.cell_models[to_cell_type.name]
            if not from_cell_model.relay:
                continue
            if to_cell_model.relay:
                report(
                    "Adding",
                    len(connectivity_set),
                    connection_model.name,
                    "connections as intermediate.",
                    level=3,
                )
                bin = intermediate_relays
                connections = connectivity_set.connections
                target = lambda c: c.to_id
            else:
                report(
                    "Adding",
                    len(connectivity_set),
                    connection_model.name,
                    "connections as terminal.",
                    level=3,
                )
                bin = terminal_relays
                connections = connectivity_set.intersections
                target = lambda c: (
                    c.to_id,
                    c.from_compartment,
                    c.to_compartment,
                    connection_model,
                )
            for id in self.scaffold.get_placement_set(from_cell_type.name).identifiers:
                if id not in bin:
                    bin[id] = []
            for connection in connections:
                fid = connection.from_id
                bin[fid].append(target(connection))

        report("Relays indexed, resolving intermediates.")

        interm_transfer = {k: [] for k in intermediate_relays.keys()}
        while len(intermediate_relays) > 0:
            intermediates_to_remove = []
            for intermediate, targets in intermediate_relays.items():
                for target in targets:
                    if target in intermediate_relays:
                        # This target of this intermediary is also an
                        # intermediary and cannot be resolved to a terminal at
                        # this point, so we wait until a next iteration where
                        # the intermediary target might have been resolved.
                        continue
                    elif target in terminal_relays:
                        # The target is a terminal relay and can be removed from
                        # our intermediary target list and its terminal targets
                        # added to our terminal target list.
                        arr = interm_transfer[intermediate]
                        assert all(
                            isinstance(t, tuple) for t in terminal_relays[target]
                        ), f"Terminal relay {lookup(target)} {target} contains non-terminal targets: {terminal_relays[target]}"
                        arr.extend(terminal_relays[target])
                        targets.remove(target)
                    else:
                        raise RelayError(
                            f"Non-relay {lookup(target)} {target} found in intermediate relay map."
                        )
                # If we have no more intermediary targets, we can be removed from
                # the intermediary relay list and be moved to the terminals.
                if not targets:
                    intermediates_to_remove.append(intermediate)
                    terminal_relays[intermediate] = interm_transfer.pop(intermediate)
            for intermediate in intermediates_to_remove:
                report(
                    "Intermediate resolved to",
                    len(terminal_relays[intermediate]),
                    "targets",
                    level=4,
                )
                intermediate_relays.pop(intermediate, None)

        report("Relays resolved.")

        # Filter out all relays to targets not on this node.
        self._relays_on = {gid: [] for gid in self.gids}
        for relay, targets in terminal_relays.items():
            assert all(
                isinstance(t, tuple) for t in terminal_relays[target]
            ), f"Terminal relay {lookup(target)} {target} contains non-terminal targets: {terminal_relays[target]}"

            for conn in targets:
                to_id = int(conn[0])
                if to_id in self.gids:
                    self._relays_on[to_id].append((relay, *conn[1:]))
        report(
            "Node",
            self.get_rank(),
            "needs to relay",
            sum(bool(relays) for relays in self._relays_on.values()),
            "relays.",
            level=4,
        )

    def prepare_devices(self):
        device_module = __import__("devices", globals(), level=1).__dict__
        for device in self.devices.values():
            device_class = "".join(x.title() for x in device.device.split("_"))
            device._bootstrap(device_module[device_class])

    def _cache_devices(self):
        self._devices_on = {gid: [] for gid in self.gids}
        for device in self.devices.values():
            targets = device.get_targets()
            for target in targets:
                self._devices_on[target].append(device)


class ArborAdapter:
    Simulation = ArborSimulation
