import abc, random, types
import numpy as np
from .helpers import ConfigurableClass, assert_attr, SortableByAfter
from .exceptions import *


class SimulationComponent(ConfigurableClass, SortableByAfter):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.simulation = None

    def get_config_node(self):
        return self.node_name + "." + self.name

    @classmethod
    def get_ordered(cls, objects):
        return objects.values()  # No sorting of simulation components required.

    def get_after(self):
        return None if not self.has_after() else self.after

    def create_after(self):
        self.after = []

    def has_after(self):
        return hasattr(self, "after")


class SimulationCell(SimulationComponent):
    def boot(self):
        super().boot()
        try:
            self.cell_type = self.scaffold.get_cell_type(self.name)
        except TypeNotFoundError:
            raise TypeNotFoundError(
                "Cell type '{}' not found in '{}', all cell models need to have the name of a cell type.".format(
                    self.name, self.get_config_node()
                )
            )

    def is_relay(self):
        return self.cell_type.relay

    @property
    def relay(self):
        return self.is_relay()


class SimulatorAdapter(ConfigurableClass):
    def __init__(self):
        super().__init__()
        self.cell_models = {}
        self.connection_models = {}
        self.devices = {}
        self.entities = {}
        self._progress_listeners = []

    def get_configuration_classes(self):
        if not hasattr(self.__class__, "simulator_name"):
            raise AttributeMissingError(
                "The SimulatorAdapter {} is missing the class attribute 'simulator_name'".format(
                    self.__class__
                )
            )
        # Check for the 'configuration_classes' class attribute
        if not hasattr(self.__class__, "configuration_classes"):
            raise AdapterError(
                "The '{}' adapter class needs to set the 'configuration_classes' class attribute to a dictionary of configurable classes (str or class).".format(
                    self.simulator_name
                )
            )
        classes = self.configuration_classes
        keys = ["cell_models", "connection_models", "devices"]
        # Check for the presence of required classes
        for requirement in keys:
            if requirement not in classes:
                raise AdapterError(
                    "{} adapter: The 'configuration_classes' dictionary requires a class under the '{}' key.".format(
                        self.simulator_name, requirement
                    )
                )
        # Test if they are all children of the ConfigurableClass class
        for class_key in keys:
            if not issubclass(classes[class_key], ConfigurableClass):
                raise AdapterError(
                    "{} adapter: The configuration class '{}' should inherit from ConfigurableClass".format(
                        self.simulator_name, class_key
                    )
                )
        return self.configuration_classes

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
        self.scaffold.report(
            "Simulated {}/{}ms".format(progression, duration), 3, ongoing=True
        )
        progress = types.SimpleNamespace(
            progression=progression, duration=duration, time=time()
        )
        for listener in self._progress_listeners:
            listener(progress)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)


class TargetsNeurons:
    neuron_targetting_types = ["local", "cylinder", "cell_type"]

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
        cell_types = [self.scaffold.get_cell_type(t) for t in self.cell_types]
        if len(cell_types) != 1:
            # Compile a list of the different cell type cells.
            target_cells = np.empty((0, 1))
            for t in cell_types:
                if t.entity:
                    ids = self.scaffold.get_entities_by_type(t.name)
                else:
                    ids = self.scaffold.get_cells_by_type(t.name)[:, 0]
                target_cells = np.vstack((target_cells, ids))
            return target_cells
        else:
            # Retrieve a single list
            t = cell_types[0]
            if t.entity:
                ids = self.scaffold.get_entities_by_type(t.name)
            else:
                ids = self.scaffold.get_cells_by_type(t.name)[:, 0]
            return ids

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

    def get_targets(self):
        """
            Return the targets of the device.
        """
        return self._get_targets()


class TargetsSections:
    def target_section(self, cell):
        if hasattr(self, "section_type"):
            return random.choice(cell.__dict__[self.section_type])
        return cell.soma[0]
