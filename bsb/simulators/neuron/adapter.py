import contextlib
import itertools
import os
import random
import time
import traceback

import errr
import numpy as np

from bsb.exceptions import AdapterError, DatasetNotFoundError, TransmitterError
from bsb.reporting import report, warn
from bsb.services import MPI
from bsb.simulation.adapter import SimulatorAdapter
from bsb.simulation.results import SimulationRecorder, SimulationResult
from bsb.storage import Chunk


class NeuronEntity:
    @classmethod
    def instantiate(cls, **kwargs):
        instance = cls()
        instance.entity = True
        for k, v in kwargs.items():
            instance.__dict__[k] = v
        return instance

    def set_reference_id(self, id):
        self.ref_id = id

    def record_soma(self):
        raise NotImplementedError("Entities do not have a soma to record.")


class SimulationData:
    def __init__(self):
        self.chunks = None
        self.cells = dict()
        self.result = None


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
    def __init__(self):
        super().__init__()
        self.engine = None
        self.network = None
        self.result = None
        self.simdata = dict()

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

            report("Load balancing", level=2)
            self.load_balance(simulation)
            self.simdata[simulation].result = SimulationResult(simulation)
            report("Load balancing", level=2)
            self.create_neurons(simulation)
            MPI.barrier()
            report("Creating transmitters", level=2)
            self.create_transmitters()
            report("", level=3)
            self.create_source_vars()
            report("Indexing relays", level=2)
            self.index_relays()
            MPI.barrier()
            report("Creating receivers", level=2)
            self.create_receivers()
            MPI.barrier()
            report("Preparing devices", level=2)
            self.prepare_devices()
            MPI.barrier()
            report("Creating devices", level=2)
            self.create_devices()
            MPI.barrier()
        except Exception:
            del self.simdata[simulation]
            raise

    def load_balance(self, simulation):
        chunk_stats = self.network.storage.get_chunk_stats()
        size = MPI.get_size()
        rank = MPI.get_rank()
        all_chunks = [Chunk.from_id(int(chunk), None) for chunk in chunk_stats.keys()]
        self.simdata[simulation].chunks = all_chunks[rank::size]

    def run(self, simulation):
        if simulation not in self.simdata:
            raise AdapterError("Simulation was not prepared")
        try:
            pc = simulator.parallel
            self.pc = pc
            pc.barrier()
            report("Simulating...", level=2)
            pc.set_maxstep(10)
            simulator.finitialize(self.initial)
            progression = 0
            self.start_progress(self.duration)
            for oi, i in self.step_progress(self.duration, 1):
                t = time.time()
                pc.psolve(i)
                pc.barrier()
                self.progress(i)
                if os.path.exists("interrupt_neuron"):
                    report("Iterrupt requested. Stopping simulation.", level=1)
                    break
            report("Finished simulation.", level=2)
        finally:
            result = self.simdata[simulation].result
            del self.simdata[simulation]
        return result

    def create_transmitters(self):
        # Concatenates all the `from` locations of all intersections together and creates
        # a network wide map of "signal origins" to NEURON parallel spike GIDs.

        # Fetch all of the connectivity sets that can be transmitters (excludes relays)
        sets = self._collect_transmitter_sets(self.connection_models.values())
        # Get the total size of all intersections
        total = sum(len(s) for s in sets)
        # Allocate an array for them
        alloc = np.empty((total, 2), dtype=int)
        ptr = 0
        for connectivity_set in sets:
            # Get the connectivity set's intersection and slice them into the array.
            inter = connectivity_set.intersections
            if not len(inter):
                continue
            alloc[ptr : (ptr + len(inter))] = [
                (i.from_id, i.from_compartment.section_id) for i in inter
            ]
            # Move up the pointer for the next slice.
            ptr += len(inter)
        unique_transmitters = np.unique(alloc, axis=0)
        self.transmitter_map = dict(zip(map(tuple, unique_transmitters), range(total)))
        tcount = 0
        try:
            for (cell_id, section_id), gid in self.transmitter_map.items():
                if cell_id in self.node_cells:
                    cell = self.cells[cell_id]
                    cell.create_transmitter(cell.sections[section_id], gid)
                    tcount += 1
        except Exception as e:
            errr.wrap(TransmitterError, e, prepend=f"[{cell_id}] ")

        report(
            f"Node {self.get_rank()} created {tcount} transmitters",
            level=3,
            all_nodes=True,
        )

    def create_source_vars(self):
        for connection_model in self.connection_models.values():
            if not connection_model.source:
                continue
            source = connection_model.source
            set = self._model_to_set(connection_model)
            for inter in set.intersections:
                cell_id = inter.from_id
                if cell_id not in self.node_cells:
                    continue
                cell = self.cells[cell_id]
                section_id = inter.from_compartment.section_id
                section = cell.sections[section_id]
                gid = self.transmitter_map[(cell_id, section_id)]
                cell.create_transmitter(cell.sections[section_id], gid, source)

    def _collect_transmitter_sets(self, models):
        sets = self._models_to_sets(models)
        return [s for s in sets if self._is_transmitter_set(s)]

    def _models_to_sets(self, models):
        return [self._model_to_set(model) for model in models]

    def _model_to_set(self, model):
        return self.scaffold.get_connectivity_set(model.name)

    def _is_transmitter_set(self, set):
        if set.is_orphan():
            return False
        name = set.connection_types[0].from_cell_types[0].name
        from_cell_model = self.cell_models[name]
        return not from_cell_model.relay

    def create_receivers(self):
        for connection_model in self.connection_models.values():
            # Get the connectivity set associated with this connection model
            connectivity_set = self.scaffold.get_connectivity_set(connection_model.name)
            from_cell_type = connectivity_set.connection_types[0].presynaptic.type
            if self.cell_models[from_cell_type.name].relay:
                continue
            from_cell_model = self.cell_models[from_cell_type.name]
            to_cell_type = connectivity_set.connection_types[0].postsynaptic.type
            to_cell_model = self.cell_models[to_cell_type.name]
            if self.cell_models[to_cell_type.name].relay:
                raise NotImplementedError("Sorry, no relays yet, only for devices")
                # Fetch cell and section from `self.relay_scheme`
                # .get_locations() should offer some insights
            else:
                synapse_types = connection_model.resolve_synapses()
                for intersection in connectivity_set.intersections:
                    if intersection.to_id in self.node_cells:
                        cell = self.cells[int(intersection.to_id)]
                        section_id = int(intersection.to_compartment.section_id)
                        section = cell.sections[section_id]
                        gid = self.transmitter_map[
                            (
                                intersection.from_id,
                                intersection.from_compartment.section_id,
                            )
                        ]
                        for synapse_type in synapse_types:
                            try:
                                cell.create_receiver(section, gid, synapse_type)
                            except Exception as e:
                                raise ScaffoldError(
                                    "[" + connection_model.name + "] " + str(e)
                                ) from None

    def create_neurons(self, simulation):
        simdata = self.simdata[simulation]
        for cell_model in simulation.cell_models.values():
            if cell_model.relay:
                continue
            ps = cell_model.cell_type.get_placement_set()
            for chunk in simdata.chunks:
                self.create_chunk_neurons(chunk, simdata, simulation, cell_model, ps)

    def _create_chunk_neurons(self, chunk, simdata, simulation, cell_model, ps):
        with ps.chunk_context(chunk):
            data = []
            for var in ("positions", "morphologies", "rotations", "additional"):
                try:
                    data.append(getattr(ps, f"load_{var}")())
                except DatasetNotFoundError:
                    data.append(itertools.repeat(None))

            with fill_parameter_data(cell_model.parameters, data):
                simdata.cells[chunk] = [
                    cell_model.create(i, *datum) for i, datum in enumerate(data)
                ]

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
