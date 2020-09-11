import abc, random
import numpy as np
from . import config
from .config import refs, types
from .helpers import SortableByAfter
from .reporting import report
from .exceptions import *
from . import plugins
from time import time


@config.node
class SimulationComponent(SortableByAfter):
    name = config.attr(key=True)

    def __init__(self, parent=None):
        if parent is None:
            self.adapter = None
        else:
            # Get the parent of the dict  that we are defined in (cell_models,
            # connections_models, device_models, ...). This grandparent is the adapter
            self.adapter = parent._config_parent
        self.simulation = None

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of simulation components required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        self.after = []

    def has_after(self):
        return hasattr(self, "after")


@config.node
class CellModel(SimulationComponent):
    cell_type = config.ref(refs.cell_type_ref, key="name")

    def is_relay(self):
        return self.cell_type.relay

    @property
    def relay(self):
        return self.is_relay()


@config.node
class ConnectionModel(SimulationComponent):
    pass


@config.node
class DeviceModel(SimulationComponent):
    pass


class ProgressEvent:
    def __init__(progression, duration, time):
        self.progression = progression
        self.duration = duration
        self.time = time


@config.pluggable(key="simulator", plugin_name="simulator adapter")
class SimulatorAdapter:
    duration = config.attr(type=float, required=True)
    cell_models = config.slot(type=CellModel, required=True)
    connection_models = config.slot(type=ConnectionModel, required=True)
    devices = config.slot(type=DeviceModel, required=True)

    @classmethod
    def __plugins__(cls):
        if not hasattr(cls, "_plugins"):
            cls._plugins = plugins.discover("adapters")
        return cls._plugins

    def __init__(self):
        self.entities = {}
        self._progress_listeners = []

    @abc.abstractmethod
    def prepare(self, hdf5, simulation_config):
        """
            This method turns a stored HDF5 network architecture and returns a runnable simulator.

            :returns: A simulator prepared to run a simulation according to the given configuration.
        """
        pass

    @abc.abstractmethod
    def simulate(self, simulator):
        """
            Start a simulation given a simulator object.
        """
        pass

    @abc.abstractmethod
    def collect_output(self, simulator):
        """
            Collect the output of a simulation that completed
        """
        pass

    def progress(self, progression, duration):
        report("Simulated {}/{}ms".format(progression, duration), level=3, ongoing=True)
        progress = ProgressEvent(progression, duration, time())
        for listener in self._progress_listeners:
            listener(progress)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)


class TargetsNeurons:
    def initialise(self, scaffold):
        super().initialise(scaffold)
        # Set targetting method
        get_targets_name = "_targets_" + self.targetting
        method = (
            getattr(self, get_targets_name) if hasattr(self, get_targets_name) else None
        )
        if not callable(method):
            raise NotImplementedError(
                "Unimplemented neuron targetting type '{}' in {}".format(
                    self.targetting, self.node_name
                )
            )
        self._get_targets = method

    def _targets_local(self):
        """
            Target all or certain cells in a spherical location.
        """
        if len(self.cell_types) != 1:
            # Compile a list of the cells and build a compound tree.
            target_cells = np.empty((0, 5))
            id_map = np.empty((0, 1))
            for t in self.cell_types:
                cells = self.scaffold.get_cells_by_type(t)
                target_cells = np.vstack((target_cells, cells[:, 2:5]))
                id_map = np.vstack((id_map, cells[:, 0]))
            tree = KDTree(target_cells)
            target_positions = target_cells
        else:
            # Retrieve the prebuilt tree from the SHDF file
            tree = self.scaffold.trees.cells.get_tree(self.cell_types[0])
            target_cells = self.scaffold.get_cells_by_type(self.cell_types[0])
            id_map = target_cells[:, 0]
            target_positions = target_cells[:, 2:5]
        # Query the tree for all the targets
        target_ids = tree.query_radius(np.array(self.origin).reshape(1, -1), self.radius)[
            0
        ].tolist()
        return id_map[target_ids]

    def _targets_cylinder(self):
        """
            Target all or certain cells within a cylinder of specified radius.
        """
        if len(self.cell_types) != 1:
            # Compile a list of the cells.
            target_cells = np.empty((0, 5))
            id_map = np.empty((0, 1))
            for t in self.cell_types:
                cells = self.scaffold.get_cells_by_type(t)
                target_cells = np.vstack((target_cells, cells[:, 2:5]))
                id_map = np.vstack((id_map, cells[:, 0]))
            target_positions = target_cells
        else:
            # Retrieve the prebuilt tree from the SHDF file
            # tree = self.scaffold.trees.cells.get_tree(self.cell_types[0])
            target_cells = self.scaffold.get_cells_by_type(self.cell_types[0])
            # id_map = target_cells[:, 0]
            target_positions = target_cells[:, 2:5]
            # Query the tree for all the targets
            center_scaffold = [
                self.scaffold.configuration.X / 2,
                self.scaffold.configuration.Z / 2,
            ]

            # Find cells falling into the cylinder volume
            target_cells_idx = np.sum(
                (target_positions[:, [0, 2]] - np.array(center_scaffold)) ** 2, axis=1
            ).__lt__(self.radius ** 2)
            cylinder_target_cells = target_cells[target_cells_idx, 0]
            cylinder_target_cells = cylinder_target_cells.astype(int)
            cylinder_target_cells = cylinder_target_cells.tolist()
            # print(id_stim)
            return cylinder_target_cells

    def _targets_cell_type(self):
        """
            Target all cells of certain cell types
        """

    def _targets_representatives(self):
        target_types = [
            cell_model.cell_type
            for cell_model in self.adapter.cell_models.values()
            if not cell_model.cell_type.relay
        ]
        if hasattr(self, "cell_types"):
            target_types = list(filter(lambda c: c.name in self.cell_types, target_types))
        target_ids = [cell_type.get_ids() for cell_type in target_types]
        representatives = [
            random.choice(type_ids) for type_ids in target_ids if len(target_ids) > 0
        ]
        return representatives

    def _targets_by_id(self):
        return self.targets

    def get_targets(self):
        """
            Return the targets of the device.
        """
        return self._get_targets()

    # Define new targetting methods above this line or they will not be registered.
    targetting_types = [s[9:] for s in vars().keys() if s.startswith("_targets_")]


@config.dynamic(
    attr_name="type", class_map={"cell_type": "bsb.simulation.CellTypeTargetting"}
)  # auto_class_map=True)
class NeuronTargetting:
    type = config.attr(type=types.in_(TargetsNeurons.targetting_types), required=True)
    cell_types = config.attr(type=types.list(type=str))
    options = config.catch_all(type=types.any())

    def __init__(self, parent):
        self.device = parent

    def boot(self):
        self.adapter = self.device.adapter if self.device is not None else None

    def get_targets(self):
        raise NotImplementedError(
            "Targetting mechanism '{}' did not implement a `get_targets` method".format(
                self.type
            )
        )


@config.node
class CellTypeTargetting(NeuronTargetting):  # , class_map_entry="cell_type"):
    """
        Targetting mechanism (use ``"type": "cell_type"``) to target all identifiers of
        certain cell types.
    """

    cell_types = config.attr(type=types.list(type=str), required=True)

    def get_targets(self):
        sets = [self.scaffold.get_placement_set(t) for t in self.cell_types]
        ids = []
        for set in sets:
            ids.extend(set.identifiers)
        return ids


class TargetsSections:
    def target_section(self, cell):
        if not hasattr(self, "section_targetting"):
            self.section_targetting = "default"
        method_name = "_section_target_" + self.section_targetting
        if not hasattr(self, method_name):
            raise Exception(
                "Unknown section targetting type '{}'".format(self.section_targetting)
            )
        return getattr(self, method_name)(cell)

    def _section_target_default(self, cell):
        if not hasattr(self, "section_count"):
            self.section_count = 1
        if hasattr(self, "section_type"):
            sections = [s for s in cell.sections if self.section_type in s.labels]
        else:
            sections = cell.soma
        return [random.choice(sections) for _ in range(self.section_count)]
