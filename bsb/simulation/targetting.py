import random, numpy as np
from ..exceptions import *


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
        cell_types = [self.scaffold.get_cell_type(t) for t in self.cell_types]
        if len(cell_types) != 1:
            # Compile a list of the different cell type cells.
            target_cells = np.array([])
            for t in cell_types:
                if t.entity:
                    ids = self.scaffold.get_entities_by_type(t.name)
                else:
                    ids = self.scaffold.get_cells_by_type(t.name)[:, 0]
                target_cells = np.hstack((target_cells, ids))
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

    def _targets_by_id(self):
        return self.targets

    def get_targets(self):
        """
        Return the targets of the device.
        """
        return self._get_targets()

    # Define new targetting methods above this line or they will not be registered.
    neuron_targetting_types = [s[9:] for s in vars().keys() if s.startswith("_targets_")]


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
        if self.section_count == "all":
            return sections
        return [random.choice(sections) for _ in range(self.section_count)]
