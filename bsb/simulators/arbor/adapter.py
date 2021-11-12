from ...simulation import (
    SimulatorAdapter,
    SimulationComponent,
    SimulationCell,
    SimulationDevice,
    TargetsNeurons,
    TargetsSections,
    SimulationResult,
    SimulationRecorder,
)
from ...reporting import report, warn
from ...exceptions import *
from ...helpers import continuity_hop, get_configurable_class
from mpi4py.MPI import COMM_WORLD as mpi
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
    import types

    # Mock missing requirements, as arbor is, like
    # all simulators, an optional dep. of the BSB.
    arbor = types.ModuleType("arbor")
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


class ArborCell(SimulationCell):
    node_name = "simulations.?.cell_models"
    default_endpoint = "comp_-1"

    def validate(self):
        self.model_class = None
        if _has_arbor and not self.relay:
            self.model_class = get_configurable_class(self.model)

    def get_description(self, gid):
        if not self.relay:
            cell_labels = self.create_labels(gid)
            cell_decor = self.create_decor(gid)
            cc = self.model_class.cable_cell(decor=cell_decor, labels=cell_labels)
            return cc
        else:
            schedule = self.get_schedule(gid)
            return arbor.spike_source_cell(self.default_endpoint, schedule)

    def get_schedule(self, gid):
        schedule = arbor.explicit_schedule([])
        for device in self.adapter._devices_on[gid]:
            pattern = device.get_pattern(gid)
            merged = pattern + schedule.events(0, float("inf"))
            schedule = arbor.explicit_schedule(merged)
        return schedule

    def create_decor(self, gid):
        decor = arbor.decor()
        self._soma_detector(decor)
        self._create_transmitters(gid, decor)
        self._create_receivers(gid, decor)
        return decor

    def create_labels(self, gid):
        labels = arbor.label_dict()

        def comp_label(comp_id):
            labels[f"comp_{comp_id}"] = "(location 0 0)"

        comps_from = self.adapter._connections_from[gid]
        comps_on = (rcv.comp_on for rcv in self.adapter._connections_on[gid])
        it.consume(comp_label(i) for i in it.chain(comps_from, comps_on))
        return labels

    def _soma_detector(self, decor):
        decor.place("(root)", arbor.spike_detector(-10), self.default_endpoint)

    def _create_transmitters(self, gid, decor):
        for comp_id in set(self.adapter._connections_from[gid]):
            decor.place("(location 0 0)", arbor.spike_detector(-10), f"comp_{comp_id}")

    def _create_receivers(self, gid, decor):
        for rcv in self.adapter._connections_on[gid]:
            decor.place(
                f'"comp_{rcv.comp_on}"', rcv.synapse, f"comp_{rcv.comp_on}_{rcv.index}"
            )


class ArborDevice(TargetsNeurons, SimulationDevice):
    node_name = "simulations.?.devices"

    def validate(self):
        pass


class ArborConnection(SimulationComponent):
    defaults = {"gap": False, "delay": 0.025}
    casts = {"delay": float, "gap": bool}

    def validate(self):
        pass

    def make_receiver(*args):
        return Receiver(*args)


class ReceiverCollection(list):
    def __init__(self):
        super().__init__()
        self._endpoint_counters = {}

    def append(self, rcv):
        endpoint = rcv.comp_on
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
        return arbor.cell_global_label(self.from_gid, f"comp_{self.comp_from}")

    def on(self):
        return arbor.cell_local_label(f"comp_{self.comp_on}_{self.index}")


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
            for start, count in continuity_hop(iter(ps.identifier_set.get_dataset()))
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
            raise GidLookupError(f"Can't find gid {gid}.")


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
        network = self._adapter.scaffold
        s = sum(
            len(ps) for ps in map(network.get_placement_set, network.get_cell_types())
        )
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

    def _is_relay(self, gid):
        return self._adapter._lookup.lookup_kind(gid) == arbor.cell_kind.spike_source

    def _from_soma(self, gid):
        return arbor.cell_global_label(gid, "soma_spike_detector")

    def _to_soma(self):
        return arbor.cell_local_label("soma_synapse")

    def probes(self, gid):
        return (
            [arbor.cable_probe_membrane_voltage("(root)")]
            if self._adapter._lookup.lookup_kind(gid) == arbor.cell_kind.cable
            else []
        )

    def _name_of(self, gid):
        return self._adapter._lookup._lookup(gid)._type.name


class ArborAdapter(SimulatorAdapter):
    simulator_name = "arbor"

    configuration_classes = {
        "cell_models": ArborCell,
        "connection_models": ArborConnection,
        "devices": ArborDevice,
    }

    casts = {
        "duration": float,
        "resolution": float,
    }

    required = ["duration"]

    defaults = {"threads": 1, "profiling": True, "resolution": 0.025}

    def validate(self):
        if self.threads == "all":
            self.threads = psutil.cpu_count(logical=False)

    def get_rank(self):
        return mpi.Get_rank()

    def broadcast(self, data, root=0):
        return mpi.bcast(data, root)

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
            context = arbor.context(arbor.proc_allocation(self.threads), mpi)
        except TypeError:
            if mpi.Get_size() > 1:
                s = mpi.Get_size()
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
        self.domain = arbor.partition_load_balance(recipe, context)
        self.gids = set(it.chain.from_iterable(g.gids for g in self.domain.groups))
        # Cache uses the domain decomposition to cache info per gid on this node. The
        # recipe functions use the cache, but luckily aren't called until
        # `arbor.simulation` and `simulation.run`.
        self._cache_connections()
        self.prepare_devices()
        self._cache_devices()
        simulation = arbor.simulation(recipe, self.domain, context)
        report("prepared simulation", level=1)
        return simulation

    def simulate(self, simulation):
        if not mpi.Get_rank():
            simulation.record(arbor.spike_recording.all)
        self.soma_voltages = {}
        for gid in self.gids:
            self.soma_voltages[gid] = simulation.sample(
                (gid, 0), arbor.regular_schedule(0.1)
            )
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
        # import plotly.graph_objs as go

        os.makedirs("results_arbor", exist_ok=True)
        if not mpi.Get_rank():
            spikes = simulation.spikes()
            spikes = np.column_stack(
                (
                    np.fromiter((l[0][0] for l in spikes), dtype=int),
                    np.fromiter((l[1] for l in spikes), dtype=int),
                )
            )
            np.savetxt("results_arbor/spikes.txt", spikes)
            # go.Figure(go.Scatter(x=spikes[:, 1], y=spikes[:, 0], mode="markers")).show()

        for gid, probe_handle in self.soma_voltages.items():
            if not (data := simulation.samples(probe_handle)):
                continue
            np.savetxt(f"results_arbor/{gid}.txt", data[0][0])

        # go.Figure(
        #     [
        #         go.Scatter(
        #             x=data[0][0][:, 0],
        #             y=data[0][0][:, 1],
        #             name=str(gid),
        #         )
        #         for gid, probe_handle in self.soma_voltages.items()
        #         if (data := simulation.samples(probe_handle))
        #     ],
        #     layout_title_text=f"Node {mpi.Get_rank()}",
        # ).show()

    def get_recipe(self):
        return ArborRecipe(self)

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
            conn_data = conn_set.get_dataset().astype(int)
            if conn_set.has_compartment_data():
                comp_data = conn_set.compartment_set.get_dataset()
            else:
                comp_data = np.ones(conn_data.shape) * -1
            for (from_gid, to_gid), (comp_from, comp_on) in zip(conn_data, comp_data):
                from_gid = int(from_gid)
                if from_gid in self._connections_from:
                    self._connections_from[from_gid].append(comp_from)
                to_gid = int(to_gid)
                if to_gid in self._connections_on:
                    self._connections_on[to_gid].append(
                        conn_model.make_receiver(from_gid, comp_from, comp_on)
                    )

    def prepare_devices(self):
        device_module = __import__("devices", globals(), level=1).__dict__
        for device in self.devices.values():
            device_class = "".join(x.title() for x in device.device.split("_"))
            device._bootstrap(device_module[device_class])
            device.initialise_targets()

    def _cache_devices(self):
        self._devices_on = {gid: [] for gid in self.gids}
        for device in self.devices.values():
            targets = device.get_targets()
            print("Caching", device.name, device)
            for target in targets:
                self._devices_on[target].append(device)
