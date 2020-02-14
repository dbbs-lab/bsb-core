from ..simulation import SimulatorAdapter, SimulationComponent, TargetsNeurons
from ..helpers import get_configurable_class
from ..models import ConnectivitySet
from ..exceptions import MissingMorphologyError
import random, os, sys


class NeuronCell(SimulationComponent):
    node_name = "simulations.?.cell_models"

    casts = {
        "record_soma": bool,
        "record_spikes": bool,
        "relay": bool,
        "parameters": dict,
    }

    defaults = {
        "record_soma": False,
        "record_spikes": False,
        "parameters": {},
        "relay": False,
        "entity": False,
    }

    required = [lambda c: "model" if "relay" not in c or not c["relay"] else None]

    def boot(self):
        self.instances = []
        if not self.relay:
            self.model_class = get_configurable_class(self.name + "_model", self.model)

    def __getitem__(self, i):
        return self.instances[i]

    def validate(self):
        pass

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        return params


class NeuronConnection(SimulationComponent):
    node_name = "simulations.?.connection_models"

    required = ["synapse"]

    def validate(self):
        pass

    def resolve_synapse(self):
        return self.synapse


class NeuronDevice(TargetsNeurons, SimulationComponent):
    node_name = "simulations.?.devices"

    device_types = [
        "spike_generator",
        "current_generator",
        "spike_recorder",
        "voltage_recorder",
    ]

    casts = {
        "radius": float,
        "origin": [float],
    }

    defaults = {}

    required = ["targetting", "device", "io"]

    def validate(self):
        if self.device not in self.__class__.device_types:
            raise ConfigurationError(
                "Unknown device '{}' for {}".format(self.device, self.get_config_node())
            )

    def get_targets(self):
        """
            Return the targets of the device.
        """
        return self._get_targets()


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


class NeuronAdapter(SimulatorAdapter):
    """
        Interface between the scaffold model and the NEURON simulator.
    """

    simulator_name = "neuron"

    configuration_classes = {
        "cell_models": NeuronCell,
        "connection_models": NeuronConnection,
        "devices": NeuronDevice,
    }

    casts = {
        "temperature": float,
        "duration": float,
        "resolution": float,
        "initial": float,
    }

    defaults = {"initial": -65.0}

    required = ["temperature", "duration", "resolution"]

    def __init__(self):
        super().__init__()
        self.cells = {}

    def validate(self):
        pass

    def prepare(self):
        from neuron import h as simulator

        self.h = simulator

        simulator.dt = self.resolution
        simulator.celsius = self.temperature
        simulator.tstop = self.duration

        self.create_neurons()
        self.connect_neurons()
        self.create_devices()
        return simulator

    def simulate(self, simulator):
        from plotly import graph_objects as go

        # Import keyboard hit function.
        sys.path.insert(0, os.path.dirname(__file__))
        from keyboard import kbhit, start_detection, stop_detection

        sys.path = sys.path[1:]
        self.scaffold.report("Simulating (press any key to interrupt)...", 2)
        simulator.finitialize(self.initial)
        progression = 0
        start_detection()
        while progression < self.duration:
            progression += 1
            simulator.continuerun(progression)
            self.scaffold.report(
                "Simulated {}/{}ms".format(progression, self.duration), 3, ongoing=True
            )
            if kbhit():
                self.scaffold.report("Key pressed. Stopping simulation.", 1)
                break
        stop_detection()
        self.scaffold.report("Finished simulation.", 2)

    def create_neurons(self):
        import glia as g

        for cell_model in self.cell_models.values():
            cell_data = self.scaffold.get_cells_by_type(cell_model.name)
            self.scaffold.report("Placing " + str(len(cell_data)) + " " + cell_model.name)
            for cell in cell_data:
                kwargs = cell_model.get_parameters()
                kwargs["position"] = cell[2:5]
                if cell_model.entity or cell_model.relay:
                    kwargs["relay"] = cell_model.relay
                    instance = NeuronEntity.instantiate(**kwargs)
                else:
                    instance = cell_model.model_class(**kwargs)
                instance.set_reference_id(cell[0])
                if cell_model.record_soma:
                    instance.record_soma()
                cell_model.instances.append(instance)
                self.cells[cell[0]] = instance

    def connect_neurons(self):
        output_handler = self.scaffold.output_formatter
        for connection_model in self.connection_models.values():
            # Get the connectivity set associated with this connection model
            connectivity_set = ConnectivitySet(output_handler, connection_model.name)
            synapse_type = connection_model.resolve_synapse()
            try:
                intersections = connectivity_set.intersections
            except MissingMorphologyError:
                intersections = connectivity_set.connections
            self.scaffold.report(
                "Connecting " + str(len(intersections)) + " " + connection_model.name, 2
            )
            # This temporarily allows non morphological connections!!! Remove later
            # Iterate over all intersections (synaptic contacts)
            for intersection in intersections:
                # Get the cells and sections of this synaptic contact
                self.connect_intersection(intersection, synapse_type)

    def connect_intersection(self, intersection, synapse_type):
        from_cell = self.cells[int(intersection.from_id)]
        to_cell = self.cells[int(intersection.to_id)]
        if from_cell.entity:
            pass
        if not hasattr(intersection, "from_compartment"):
            from_section = from_cell.axon[-1]
            to_section = random.choice(to_cell.dendrites)
        else:
            from_section_id = intersection.from_compartment.section_id
            to_section_id = intersection.to_compartment.section_id
            from_section = from_cell.sections[from_section_id]
            to_section = to_cell.sections[to_section_id]
            if self.scaffold.configuration.verbosity >= 4:
                self.scaffold.report(
                    "Connecting "
                    + str(int(from_cell.ref_id))
                    + " from section "
                    + str(from_section_id)
                    + " ({}) ".format(",".join(from_section.labels))
                    + "to "
                    + str(int(to_cell.ref_id))
                    + " on section "
                    + str(to_section_id)
                    + " ({})".format(",".join(to_section.labels))
                    + " creating a {} synapse".format(synapse_type),
                    4,
                )
        to_cell.connect(from_cell, from_section, to_section, synapse_type)

    def create_devices(self):
        pass
