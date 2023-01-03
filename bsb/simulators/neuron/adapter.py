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
            report("Creating devices", level=2)
            self.create_devices(simulation)
            MPI.barrier()
        except:
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
        try:
            report("Simulating...", level=2)
            pc = self.engine.ParallelContext()
            pc.set_maxstep(10)
            self.engine.finitialize(self.initial)
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

    def create_neurons(self, simulation):
        simdata = self.simdata[simulation]
        offset = 0
        for cell_model in sorted(simulation.cell_models.values()):
            ps = cell_model.cell_type.get_placement_set()
            simdata.cid_offsets[cell_model.cell_type] = offset
            with ps.chunk_context(simdata.chunks):
                self._create_population(simdata, cell_model, ps, offset)
            offset += len(ps)

    def create_connections(self, simulation):
        simdata = self.simdata[simulation]
        self._allocate_transmitters(simulation)
        for conn_model in simulation.connection_models.values():
            cs = simulation.scaffold.get_connectivity_set(conn_model.name)
            with fill_parameter_data(conn_model.parameters, []):
                simdata.connections[conn_model] = conn_model.create_connections(
                    simulation, simdata, cs
                )

    def create_devices(self, simulation):
        simdata = self.simdata[simulation]
        for device_model in simulation.devices:
            device_model.implement(simdata.results, simdata.cells, simdata.connections)

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
                cid = offset + id
                instance.id = cid
                instance.model = cell_model
                simdata.cells[cid] = instance

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
