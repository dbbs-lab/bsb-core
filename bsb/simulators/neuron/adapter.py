from ...simulation import (
    Simulation,
    SimulationRecorder,
    CellModel,
    ConnectionModel,
    DeviceModel,
    NeuronTargetting,
)
from ... import config
from ...config import types
from ...reporting import report, warn
from ...exceptions import *
import random, os, sys
import numpy as np
import traceback
import errr
import time


@config.node
class NeuronCell(CellModel):
    model = config.attr(
        type=types.class_(), required=lambda s: not ("relay" in s and s["relay"])
    )
    record_soma = config.attr(default=False)
    record_spikes = config.attr(default=False)
    entity = config.attr(default=False)

    def boot(self):
        super().boot()
        self.instances = []

    def __getitem__(self, i):
        return self.instances[i]

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        return params


_str_list = types.list(type=str)


@config.node
class NeuronConnection(ConnectionModel):
    synapse = config.attr(
        type=types.or_(types.dict(type=_str_list), _str_list), required=True
    )
    source = config.attr(type=str, default=None)

    def resolve_synapses(self):
        return self.synapses


@config.dynamic(attr_name="device", type=types.in_classmap(), auto_classmap=True)
class NeuronDevice(DeviceModel):
    radius = config.attr(type=float)
    origin = config.attr(type=types.list(type=float, size=3))
    targetting = config.attr(type=NeuronTargetting, required=True)
    io = config.attr(type=types.in_(["input", "output"]), required=True)

    def create_patterns(self):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `create_patterns` function."
        )

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `get_pattern` function."
        )

    def implement(self, target, location):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )

    def get_locations(self, target):
        locations = []
        if target in self.adapter.relay_scheme:
            for cell_id, section_id, connection in self.adapter.relay_scheme[target]:
                if cell_id not in self.adapter.node_cells:
                    continue
                cell = self.adapter.cells[cell_id]
                section = cell.sections[section_id]
                locations.append(TargetLocation(cell, section, connection))
        elif target in self.adapter.node_cells:
            try:
                cell = self.adapter.cells[target]
            except KeyError:
                raise DeviceConnectionError(
                    "Missing cell {} on node {} while trying to implement device '{}'. This can occur if the cell was placed in the network but not represented with a model in the simulation config.".format(
                        target, self.adapter.get_rank(), self.name
                    )
                )
            sections = self.target_section(cell)
            locations.extend(TargetLocation(cell, section) for section in sections)
        return locations


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


@config.node
class NeuronSimulation(Simulation):
    """
    Interface between the scaffold model and the NEURON simulator.
    """

    simulator_name = "neuron"
    cell_models = config.dict(type=NeuronCell, required=True)
    connection_models = config.dict(type=NeuronConnection, required=True)
    devices = config.dict(type=NeuronDevice, required=True)
    resolution = config.attr(type=float, default=1.0)
    initial = config.attr(type=float, default=-65.0)
    temperature = config.attr(type=float, required=True)

    def __init__(self):
        self.cells = {}
        self._next_gid = 0
        self.transmitter_map = {}

    def validate(self):
        pass

    def validate_prepare(self):
        for connection_model in self.connection_models.values():
            # Get the connectivity set associated with this connection model
            connectivity_set = self.scaffold.get_connectivity_set(connection_model.name)
            from_type = connectivity_set.connection_types[0].presynaptic.type
            to_type = connectivity_set.connection_types[0].postsynaptic.type
            from_cell_model = self.cell_models[from_type.name]
            to_cell_model = self.cell_models[to_type.name]
            if (
                from_type.entity
                or from_cell_model.relay
                or to_type.entity
                or to_cell_model.relay
            ):
                continue
            if not connectivity_set.compartment_set.exists():
                raise IntersectionDataNotFoundError(
                    "No intersection data found for '{}'".format(connection_model.name)
                )

    def get_rank(self):
        return self.h.parallel.id()

    def get_size(self):
        return self.h.parallel.nhost()

    def broadcast(self, data, root=0):
        from patch import p

        return p.parallel.broadcast(data, root=root)

    def prepare(self):
        from patch import p as simulator
        from time import time

        report("Preparing simulation", level=3)

        self.validate_prepare()
        self.h = simulator

        simulator.dt = self.resolution
        simulator.celsius = self.temperature
        simulator.tstop = self.duration

        t = t0 = time()
        self.load_balance()
        report(
            "Load balancing on node",
            self.get_rank(),
            "took",
            round(time() - t, 2),
            "seconds",
            all_nodes=True,
        )
        t = time()
        self.init_result()
        self.create_neurons()
        t = time() - t
        simulator.parallel.barrier()
        report(
            "Cell creation on node",
            self.get_rank(),
            "took",
            round(t, 2),
            "seconds",
            all_nodes=True,
        )
        t = time()
        self.create_transmitters()
        self.create_source_vars()
        report(
            "Transmitter creation on node",
            self.get_rank(),
            "took",
            round(time() - t, 2),
            "seconds",
            all_nodes=True,
        )
        self.index_relays()
        simulator.parallel.barrier()
        t = time()
        self.create_receivers()
        t = time() - t
        report(
            "Receiver creation on node",
            self.get_rank(),
            "took",
            round(t, 2),
            "seconds",
            all_nodes=True,
        )
        simulator.parallel.barrier()
        t = time()
        self.prepare_devices()
        t = time() - t
        report(
            "Device preparation on node",
            self.get_rank(),
            "took",
            round(t, 2),
            "seconds",
            all_nodes=True,
        )
        simulator.parallel.barrier()
        t = time()
        self.create_devices()
        t = time() - t
        report(
            "Device creation on node",
            self.get_rank(),
            "took",
            round(t, 2),
            "seconds",
            all_nodes=True,
        )
        report("Simulator preparation took", round(time() - t0, 2), "seconds")
        return simulator

    def init_result(self):
        self.result = SimulationResult()
        if self.get_rank() == 0:
            # Record the time
            self.h.time
            self.result.create_recorder(
                lambda: tuple(["time"]),
                lambda: np.array(self.h.time),
                lambda: {"resolution": self.resolution, "duration": self.duration},
            )

    def load_balance(self):
        rank = self.get_rank()
        size = self.get_size()
        self.cell_total = self.scaffold.get_cell_total()
        # Do a lazy round robin for now.
        self.node_cells = set(range(rank, self.cell_total, size))

    def simulate(self, simulator):
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots

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

    def collect_output(self, simulator):
        import h5py, time

        timestamp = str(time.time()).split(".")[0] + str(random.random()).split(".")[1]
        timestamp = self.pc.broadcast(timestamp)
        result_path = "results_" + self.name + "_" + timestamp + ".hdf5"
        for node in range(self.get_size()):
            self.pc.barrier()
            if node == self.get_rank():
                report("Node", self.get_rank(), "is writing", level=2, all_nodes=True)
                with h5py.File(result_path, "a") as f:
                    f.attrs["configuration_string"] = self.scaffold.configuration._raw
                    for path, data, meta in self.result.safe_collect():
                        try:
                            path = "/".join(path)
                            if path in f:
                                data = np.concatenate((f[path][()], data))
                                del f[path]
                            d = f.create_dataset(path, data=data)
                            for k, v in meta.items():
                                d.attrs[k] = v
                        except Exception as e:
                            if not isinstance(data, np.ndarray):
                                warn(
                                    "Recorder {} numpy.ndarray expected, got {}".format(
                                        path, type(data)
                                    )
                                )
                            else:
                                warn(
                                    "Recorder {} processing errored out: {}\n\n{}".format(
                                        path,
                                        "{} {}".format(data.dtype, data.shape),
                                        traceback.format_exc(),
                                    )
                                )
            self.pc.barrier()
        return result_path

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

    def create_neurons(self):
        for cell_model in self.cell_models.values():
            cell_positions = None
            if self.scaffold.configuration.get_cell_type(cell_model.name).entity:
                cell_data = self.scaffold.get_entities_by_type(cell_model.name)
                cell_data = np.column_stack((cell_data, np.zeros((len(cell_data), 4))))
            else:
                cell_data = self.scaffold.get_cells_by_type(cell_model.name)
            report("Placing " + str(len(cell_data)) + " " + cell_model.name)
            for cell in cell_data:
                cell_id = int(cell[0])
                if not cell_id in self.node_cells:
                    continue
                kwargs = cell_model.get_parameters()
                kwargs["position"] = cell[2:5]
                if cell_model.entity or cell_model.relay:
                    kwargs["relay"] = cell_model.relay
                    instance = NeuronEntity.instantiate(**kwargs)
                else:
                    instance = cell_model.model_class(**kwargs)
                instance.set_reference_id(cell_id)
                instance.cell_model = cell_model
                if cell_model.record_soma:
                    self.register_cell_recorder(instance, instance.record_soma())
                if cell_model.record_spikes:
                    spike_nc = self.h.NetCon(instance.soma[0], None)
                    spike_nc.threshold = -20
                    spike_recorder = spike_nc.record()
                    self.register_spike_recorder(instance, spike_recorder)
                cell_model.instances.append(instance)
                self.cells[cell_id] = instance
        report(
            f"Node {self.get_rank()} created {len(self.cells)} cells",
            level=2,
            all_nodes=True,
        )

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
        self.relay_scheme = {}
        for relay, targets in terminal_relays.items():
            assert all(
                isinstance(t, tuple) for t in terminal_relays[target]
            ), f"Terminal relay {lookup(target)} {target} contains non-terminal targets: {terminal_relays[target]}"
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


class NeuronAdapter:
    Simulation = NeuronSimulation


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


class TargetLocation:
    def __init__(self, cell, section, connection=None):
        self.cell = cell
        self.section = section
        self.connection = connection

    def get_synapses(self):
        return self.connection and self.connection.synapses


class SpikeRecorder(LocationRecorder):
    def get_data(self):
        recording = np.array(list(self.recorder))
        return np.column_stack((np.ones(recording.shape) * self.id, recording))
