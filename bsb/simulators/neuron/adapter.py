import contextlib
import itertools
import os
import time
from functools import cache

import numpy as np
import typing

from bsb.exceptions import AdapterError, DatasetNotFoundError, TransmitterError
from bsb.reporting import report
from bsb.services import MPI
from bsb.simulation.adapter import SimulatorAdapter
from bsb.simulation.results import SimulationRecorder, SimulationResult
from bsb.storage import Chunk

if typing.TYPE_CHECKING:
    from bsb.simulation.simulation import Simulation


class SimulationData:
    def __init__(self):
        self.chunks = None
        self.cells = dict()
        self.cid_offsets = dict()
        self.connections = dict()
        self.first_gid: int = None
        self.result: SimulationResult = None


@contextlib.contextmanager
def fill_parameter_data(parameters, data):
    for param in parameters:
        if hasattr(param, "load_data"):
            param.load_data(*data)
    yield
    for param in parameters:
        if hasattr(param, "load_data"):
            param.drop_data()


class NeuronAdapter(SimulatorAdapter):
    initial = -65

    def __init__(self):
        super().__init__()
        self.engine = None
        self.network = None
        self.result = None
        self.simdata = dict()
        self.next_gid = 0

    def prepare(self, simulation, comm=None):
        if self.engine is None:
            from patch import p as engine

            self.engine = engine

        self.simdata[simulation] = SimulationData()
        try:
            report("Preparing simulation", level=2)
            engine.dt = simulation.resolution
            engine.celsius = simulation.temperature
            engine.tstop = simulation.duration
            simdata = self.simdata[simulation]

            report("Load balancing", level=2)
            self.load_balance(simulation)
            simdata.result = SimulationResult(simulation)
            report("Load balancing", level=2)
            self.create_neurons(simulation)
            report("Creating transmitters", level=2)
            self.create_connections(simulation)
            report("", level=3)
            # self.create_source_vars()
            report("Indexing relays", level=2)
            # self.index_relays()
            MPI.barrier()
            report("Creating receivers", level=2)
            # self.create_receivers()
            MPI.barrier()
            report("Preparing devices", level=2)
            # self.prepare_devices()
            MPI.barrier()
            report("Creating devices", level=2)
            # self.create_devices()
            MPI.barrier()
        except Exception:
            del self.simdata[simulation]
            raise
        return self.simdata[simulation]

    def load_balance(self, simulation):
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

    def run(self, simulation: "Simulation"):
        if simulation not in self.simdata:
            raise AdapterError("Simulation was not prepared")
        data = self.simdata[simulation]
        try:
            report("Simulating...", level=2)
            pc = self.engine.ParallelContext()
            pc.set_maxstep(10)
            self.engine.finitialize(self.initial)
            progression = 0
            simulation.start_progress(simulation.duration)
            for oi, i in simulation.step_progress(simulation.duration, 1):
                t = time.time()
                pc.psolve(i)
                simulation.progress(i)
                if os.path.exists("interrupt_neuron"):
                    report("Iterrupt requested. Stopping simulation.", level=1)
                    break
            report("Finished simulation.", level=2)
        finally:
            result = self.simdata[simulation].result
            del self.simdata[simulation]

        return result

    def collect(self, simulation: "Simulation", data: SimulationData):
        data.result.flush()
        return data.result

    def create_connections(self, simulation):
        # Offset based deterministic connection to GID algorithm.
        simdata = self.simdata[simulation]
        self._allocate_transmitters(simulation)
        for conn_model in simulation.connection_models.values():
            cs = simulation.scaffold.get_connectivity_set(conn_model.name)
            with fill_parameter_data(conn_model.parameters, []):
                conn_model.create_connections(simulation, simdata, cs)

    @cache
    def _allocate_transmitters(self, simulation):
        simdata = self.simdata[simulation]
        first = self.next_gid
        chunk_stats = simulation.scaffold.storage.get_chunk_stats()
        max_trans = sum(stats["connections"]["out"] for stats in chunk_stats.values())
        report(
            f"Node {MPI.get_rank()} allocated GIDs {self.next_gid} to {max_trans}",
            level=3,
            all_nodes=True,
        )
        simdata.alloc = (first, self.next_gid)
        simdata.transmap = self._map_transmitters(simulation, simdata)

    @cache
    def _map_transmitters(self, simulation, simdata):
        blocks = []
        for cm, cs in simulation.get_connectivity_sets().items():
            pre, _ = cs.load_connections().as_globals().all()
            pre[:, 0] += simdata.cid_offsets[cs.pre_type]
            blocks.append(pre[:, :2])
        return {
            tuple(loc): gid + simdata.alloc[0]
            for gid, loc in zip(
                itertools.count(), np.unique(np.concatenate(blocks), axis=0)
            )
        }

    def create_neurons(self, simulation):
        simdata = self.simdata[simulation]
        offset = 0
        for cell_model in sorted(simulation.cell_models.values()):
            ps = cell_model.cell_type.get_placement_set()
            simdata.cid_offsets[cell_model.cell_type] = offset
            with ps.chunk_context(simdata.chunks):
                self._create_population(simdata, cell_model, ps, offset)
            offset += len(ps)

    def _create_population(self, simdata, cell_model, ps, offset):
        data = []
        for var in ("positions", "morphologies", "rotations", "additional"):
            try:
                data.append(getattr(ps, f"load_{var}")())
            except DatasetNotFoundError:
                data.append(itertools.repeat(None))
        with fill_parameter_data(cell_model.parameters, data):
            instances = cell_model.create_instances(len(ps), *data)
            for id, instance in zip(ps.load_ids(), instances):
                gid = offset + id
                instance.id = gid
                simdata.cells[gid] = instance

    def prepare_devices(self):
        device_module = __import__("devices", globals(), level=1)
        for device in self.devices.values():
            # CamelCase the snake_case to obtain the class name
            device_class = "".join(x.title() for x in device.device.split("_"))
            device._bootstrap(device_module.__dict__[device_class])

    def create_devices(self):
        for device in self.devices.values():
            if self.get_rank() == 0:
                # Have root 0 prepare the possibly random targets.
                targets = device.get_targets()
            else:
                targets = None
            # Broadcast to make sure all the nodes have the same targets for each device.
            targets = self.broadcast(targets, root=0)
            for target in targets:
                for location in device.get_locations(target):
                    device.implement(target, location)

    def index_relays(self):
        report("Indexing relays.")
        terminal_relays = {}
        intermediate_relays = {}
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
            connectivity_set = self.scaffold.get_connectivity_set(connection_model.name)
            from_cell_type = connectivity_set.connection_types[0].presynaptic.type
            from_cell_model = self.cell_models[from_cell_type.name]
            to_cell_type = connectivity_set.connection_types[0].postsynaptic.type
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
                    c.to_compartment.section_id,
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
                        ), (
                            f"Terminal relay {lookup(target)} {target} contains "
                            f"non-terminal targets: {terminal_relays[target]}"
                        )
                        arr.extend(terminal_relays[target])
                        targets.remove(target)
                    else:
                        raise RelayError(
                            f"Non-relay {lookup(target)} {target} found in intermediate "
                            f"relay map."
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
        self.relay_scheme = {}
        for relay, targets in terminal_relays.items():
            assert all(isinstance(t, tuple) for t in terminal_relays[target]), (
                f"Terminal relay {lookup(target)} {target} contains non-terminal "
                f"targets: {terminal_relays[target]}"
            )
            node_targets = [x for x in targets if int(x[0]) in self.node_cells]
            self.relay_scheme[relay] = node_targets
        report(
            "Node",
            self.get_rank(),
            "needs to relay",
            len(self.relay_scheme),
            "relays.",
            level=4,
        )

    def register_recorder(
        self, group, cell, recorder, time_recorder=None, section=None, x=None, meta=None
    ):
        # Store the recorder so its output can be collected after the simulation.
        self.result.add(
            LocationRecorder(group, cell, recorder, time_recorder, section, x, meta)
        )

    def register_cell_recorder(self, cell, recorder):
        self.result.add(LocationRecorder("soma_voltages", cell, recorder))

    def register_spike_recorder(self, cell, recorder):
        self.result.add(SpikeRecorder("soma_spikes", cell, recorder))


class LocationRecorder(SimulationRecorder):
    def __init__(
        self, group, cell, recorder, time_recorder=None, section=None, x=None, meta=None
    ):
        # Collect metadata
        meta = meta or {}
        meta["cell_id"] = cell.ref_id
        meta["label"] = cell.cell_model.name
        if hasattr(cell.cell_model.cell_type, "plotting"):
            # Pass plotting info along
            meta["color"] = cell.cell_model.cell_type.plotting.color
            meta["display_label"] = cell.cell_model.cell_type.plotting.label
        self.group = group
        self.meta = meta
        self.recorder = recorder
        self.time_recorder = time_recorder
        self.section = section
        self.x = x
        # Compose the tag: `cell.section_name(x)`
        self.id = cell.ref_id
        self.tag = str(cell.ref_id)
        if section is not None:
            meta["section"] = cell.sections.index(section)
            self.tag += "." + section.name().split(".")[-1]
            if x is not None:
                self.tag += "(" + str(x) + ")"

    def get_path(self):
        return ("recorders", self.group, self.tag)

    def get_data(self):
        if self.time_recorder:
            return np.column_stack((list(self.recorder), list(self.time_recorder)))
        else:
            return np.array(list(self.recorder))

    def get_meta(self):
        return self.meta


class SpikeRecorder(LocationRecorder):
    def get_data(self):
        recording = np.array(list(self.recorder))
        return np.column_stack((np.ones(recording.shape) * self.id, recording))
